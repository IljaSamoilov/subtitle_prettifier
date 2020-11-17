import string
from typing import List, Dict

import webvtt
from estnltk import Text
from nltk import edit_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def is_word_in_caption(capt: webvtt.Caption, word_dict: dict) -> bool:
    return capt.end_in_seconds > word_dict['end']


def get_similarity(this_caption: webvtt.Caption, word_dict: List[Dict]) -> (float, float):
    subtitile_lemmas = create_lemmatized_string(this_caption.text)

    generated = ' '.join([x['word'] for x in word_dict])
    generated_text_lemmas = create_lemmatized_string(generated)
    levenshtein_distance = edit_distance(subtitile_lemmas, generated_text_lemmas)
    arr = [subtitile_lemmas, generated_text_lemmas]

    # penalty = calculate_penalty(subtitile_lemmas, generated_text_lemmas)

    vectorizer = CountVectorizer().fit_transform(arr)
    vectors = vectorizer.toarray()

    vec1 = vectors[0].reshape(1, -1)
    vec2 = vectors[1].reshape(1, -1)

    # return cosine_similarity(vec1, vec2)[0][0] * penalty, levenshtein_distance / (penalty if penalty != 0 else 0.01)
    return cosine_similarity(vec1, vec2)[0][0], levenshtein_distance


def create_lemmatized_string(generated: string) -> string:
    generated = Text(clean_string(generated))
    generated.tag_layer(['morph_analysis'])
    gen_lemmas = [item[0] for sublist in generated.lemma.amb_attr_tuple_list for item in sublist]
    gen_lemmas = ' '.join(gen_lemmas)
    return gen_lemmas


def clean_string(text):
    text = text.replace('\n', " ")
    cleaned_text = ''.join([word for word in text if word not in string.punctuation]).lower()
    return cleaned_text
#     word_array = cleaned_text.split(" ")
#     # word_array.sort()
#     # return ' '.join(word_array)