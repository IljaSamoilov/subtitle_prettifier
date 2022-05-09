import string
from typing import List, Dict

import webvtt
from estnltk import Text
from nltk import edit_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

special_words = {'protsent': '%',
                 'protsend': '%',
                 'ja': '&',
                 'punkt': '.',
                 'null': '0',
                 'üks': '1',
                 'kaks': '2',
                 'kolm': '3',
                 'neli': '4',
                 'viis': '5',
                 'kuus': '6',
                 'seitse': '7',
                 'kaheksa': '8',
                 'üheksa': '9',
                 }
punctuation = r"""!"#$'()*+,-./:;<=>?@[\]^_`{|}~"""

def is_word_in_caption(capt: webvtt.Caption, word_dict: dict) -> bool:
    return capt.end_in_seconds > word_dict['end']


def get_similarity(this_caption: webvtt.Caption, word_dict: List[Dict]) -> (float, float):
    subtitile_lemmas = create_lemmatized_string(this_caption.text)

    generated = ' '.join([x['word'] for x in word_dict])
    generated_text_lemmas = create_lemmatized_string(generated)

    generated_with_replacements = replace_symbols(generated)

    if generated_with_replacements != generated and any(word in subtitile_lemmas for word in special_words.keys()):
        lemmas_with_replacements = create_lemmatized_string(generated_with_replacements)
        levenshtein_distance = min(edit_distance(subtitile_lemmas, generated_text_lemmas),
                                   edit_distance(subtitile_lemmas, lemmas_with_replacements))
    else:
        levenshtein_distance = edit_distance(subtitile_lemmas, generated_text_lemmas)

    arr = [subtitile_lemmas, generated_text_lemmas]

    # penalty = calculate_penalty(subtitile_lemmas, generated_text_lemmas)

    # vectorizer = CountVectorizer().fit_transform(arr)
    # vectors = vectorizer.toarray()
    #
    # vec1 = vectors[0].reshape(1, -1)
    # vec2 = vectors[1].reshape(1, -1)

    # return cosine_similarity(vec1, vec2)[0][0] * penalty, levenshtein_distance / (penalty if penalty != 0 else 0.01)
    return 0, levenshtein_distance

def get_ratio(this_caption: webvtt.Caption, word_dict: List[Dict]) -> (float, float):
    subtitile_lemmas = create_lemmatized_string(this_caption.text)

    generated = ' '.join([x['word'] for x in word_dict])
    generated_text_lemmas = create_lemmatized_string(generated)

    generated_with_replacements = replace_symbols(generated)

    if generated_with_replacements != generated and any(word in subtitile_lemmas for word in special_words.keys()):
        lemmas_with_replacements = create_lemmatized_string(generated_with_replacements)
        ratio = min(fuzz.ratio(subtitile_lemmas, generated_text_lemmas),
                    fuzz.ratio(subtitile_lemmas, lemmas_with_replacements))
    else:
        ratio = fuzz.ratio(subtitile_lemmas, generated_text_lemmas)

    # return cosine_similarity(vec1, vec2)[0][0] * penalty, levenshtein_distance / (penalty if penalty != 0 else 0.01)
    return 0, ratio


def create_lemmatized_string(generated: string) -> string:
    generated = Text(clean_string(generated))
    generated.tag_layer(['morph_analysis'])
    gen_lemmas = [sublist[0][0] for sublist in generated.lemma.amb_attr_tuple_list]
    gen_lemmas = ' '.join(gen_lemmas)
    return gen_lemmas


def replace_symbols(text: string) -> string:
    replaced_text = text
    for word, replacement in special_words.items():
        replaced_text = replaced_text.replace(word, replacement)
    return replaced_text


def clean_string(text):
    text = text.replace('\n', " ")
    cleaned_text = ''.join([word for word in text if word not in punctuation]).lower()
    return cleaned_text
#     word_array = cleaned_text.split(" ")
#     # word_array.sort()
#     # return ' '.join(word_array)
