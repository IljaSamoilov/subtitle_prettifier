import json
from typing import List, Dict

import webvtt

from sections import SubtitlePairWords

# from estnltk.taggers import VabamorfTagger
import os

import numpy as np

from utility import is_word_in_caption, get_similarity


def generate_results(pairs: List[SubtitlePairWords], name: str):
    result_file_name = f'{name}-result-sorted-lemmas.txt'
    if os.path.exists(result_file_name):
        os.remove(result_file_name)
    with open(result_file_name, encoding='utf-8', errors='ignore', mode="a+") as result_file:
        for pair in pairs:
            result_file.write(pair.__str__())
            result_file.write("\n\n")
        mean = np.mean([item.similarity for item in pairs])
        result_file.write(f"Mean similarity: {mean}")


def process_subtitles(file_name: str, use_levishtein=False) -> List[SubtitlePairWords]:
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

        cos_similarity, lev_distance = get_similarity(caption, words_in_captions)
        similarity = lev_distance if use_levishtein else cos_similarity
        if i < len(words):
            i, similarity, words_in_captions = try_more_or_less_words(caption, i, similarity, words, words_in_captions, use_levishtein)

        pairs.append(SubtitlePairWords(words_in_captions, caption, similarity))

    return pairs


def try_more_or_less_words(caption, i, similarity, words, words_in_captions, use_levishtein):
    if i >= len(words):
        return i, similarity, words_in_captions

    with_next_word = words_in_captions + [words[i]]
    one_less_word = words_in_captions[:-1]

    similarity_with_next_word, lev_distance1 = get_similarity(caption, with_next_word)
    similarity_one_less_word, lev_distance2 = get_similarity(caption, one_less_word)
    if use_levishtein:
        if lev_distance1 < similarity or lev_distance2 < similarity:
            if lev_distance1 < lev_distance2:
                i += 1
                words_in_captions = with_next_word
                similarity = lev_distance1
                return try_more_or_less_words(caption, i, similarity, words, words_in_captions, use_levishtein)
            else:
                i -= 1
                words_in_captions = one_less_word
                similarity = lev_distance2
                return try_more_or_less_words(caption, i, similarity, words, words_in_captions, use_levishtein)
        return i, similarity, words_in_captions
    else:
        if similarity_with_next_word > similarity or similarity_one_less_word > similarity:
            if similarity_with_next_word > similarity_one_less_word:
                i += 1
                words_in_captions = with_next_word
                similarity = similarity_with_next_word
                return try_more_or_less_words(caption, i, similarity, words, words_in_captions, use_levishtein)
            else:
                i -= 1
                words_in_captions = one_less_word
                similarity = similarity_one_less_word
                return try_more_or_less_words(caption, i, similarity, words, words_in_captions, use_levishtein)
        return i, similarity, words_in_captions


def calculate_penalty(first: str, second: str) -> float:
    difference = abs(first.__len__() - second.__len__())
    max_str_len = max(first.__len__(), second.__len__())
    return (max_str_len - difference) / max_str_len


if __name__ == '__main__':
    files_no_ext = set([os.path.splitext(f)[0] for f in os.listdir('data/')])

    for file_name in files_no_ext:
        if file_name.startswith("."):
            continue
        result_pairs = process_subtitles(file_name, use_levishtein=True)
        generate_results(result_pairs, file_name)
