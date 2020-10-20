import json
from typing import List, Dict, Tuple

import webvtt

from sections import SubtitlePair, TranscriptSection, SubtitlePairWords

import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from estnltk import Text
from estnltk.taggers import VabamorfTagger
import os

import numpy as np


import Levenshtein

def clean_string(text):
    text = text.replace('\n', " ")
    return ''.join([word for word in text if word not in string.punctuation]).lower()


def is_word_in_caption(capt: webvtt.Caption, word_dict: dict) -> bool:
    return capt.end_in_seconds > word_dict['end']


def get_cosine_similarity(this_caption: webvtt.Caption, word_dict: List[Dict]) -> float:
    real_lemmas = create_lemmatized_string(this_caption.text)

    generated = ' '.join([x['word'] for x in word_dict])
    gen_lemmas = create_lemmatized_string(generated)

    arr = [real_lemmas, gen_lemmas]

    vectorizer = CountVectorizer().fit_transform(arr)
    vectors = vectorizer.toarray()

    vec1 = vectors[0].reshape(1, -1)
    vec2 = vectors[1].reshape(1, -1)

    return cosine_similarity(vec1, vec2)[0][0]


def create_lemmatized_string(generated):
    generated = Text(clean_string(generated))
    generated.tag_layer(['morph_analysis'])
    gen_lemmas = [item[0] for sublist in generated.lemma.amb_attr_tuple_list for item in sublist]
    gen_lemmas = ' '.join(gen_lemmas)
    return gen_lemmas


def generate_results(pairs: List[SubtitlePairWords], name: str):
    result_file_name = f'{name}-result-try-more-or-less.txt'
    if os.path.exists(result_file_name):
        os.remove(result_file_name)
    with open(result_file_name, encoding='utf-8', errors='ignore', mode="a+") as result_file:
        for pair in pairs:
            result_file.write(pair.__str__())
            result_file.write("\n\n")
        mean = np.mean([item.similarity for item in pairs])
        result_file.write(f"Mean similarity: {mean}")

def process_subtitles(file_name: str) -> List[SubtitlePairWords]:
    subtitles = webvtt.read(f'data/{file_name}.vtt')

    with open(f'data/{file_name}.json', encoding='utf-8', errors='ignore') as fh:
        json_text = json.load(fh)

    generated_subtitles = [x['turns'] for x in json_text['sections'] if (x['type'] == "speech" and "turns" in x.keys())]
    generated_subtitles = [item for sublist in generated_subtitles for item in sublist]

    i = 0
    pairs: List[SubtitlePairWords] = []

    words_arrays = list(map(lambda x: x['words'], generated_subtitles))
    words = [item for sublist in words_arrays for item in sublist]

    for caption in subtitles.captions:

        words_in_captions: List[Dict] = []

        while i < len(words) and is_word_in_caption(caption, words[i]):
            words_in_captions.append(words[i])
            i += 1

        similarity: float = get_cosine_similarity(caption, words_in_captions)

        if i < len(words):
            i, similarity, words_in_captions = try_more_or_less_words(caption, i, similarity, words, words_in_captions)

        pairs.append(SubtitlePairWords(words_in_captions, caption, similarity))

    return pairs


def try_more_or_less_words(caption, i, similarity, words, words_in_captions):
    if i >= len(words):
        return i, similarity, words_in_captions

    with_next_word = words_in_captions + [words[i]]
    one_less_word = words_in_captions[:-1]

    similarity_with_next_word = get_cosine_similarity(caption, with_next_word)
    similarity_one_less_word = get_cosine_similarity(caption, one_less_word)

    if similarity_with_next_word > similarity or similarity_one_less_word > similarity:
        if similarity_with_next_word > similarity_one_less_word:
            i += 1
            words_in_captions = with_next_word
            similarity = similarity_with_next_word
            return try_more_or_less_words(caption, i, similarity, words, words_in_captions)
        else:
            i -= 1
            words_in_captions = one_less_word
            similarity = similarity_one_less_word
            return try_more_or_less_words(caption, i, similarity, words, words_in_captions)
    return i, similarity, words_in_captions


if __name__ == '__main__':
    files_no_ext = set([os.path.splitext(f)[0] for f in os.listdir('data/')])

    for file_name in files_no_ext:
        if file_name.startswith("."):
            continue
        result_pairs = process_subtitles(file_name)
        generate_results(result_pairs, file_name)