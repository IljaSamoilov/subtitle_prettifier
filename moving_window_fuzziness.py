import json
from typing import List, Dict, Tuple
import os

import webvtt
import numpy as np

from sections import SubtitlePair, TranscriptSection, SubtitlePairWords, SubtitlePairWordsEncoder

from utility import is_word_in_caption, get_ratio

import copy


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

        cos_similarity, lev_distance = get_ratio(caption, words_in_captions)
        similarity = lev_distance if use_levishtein else cos_similarity

        pairs.append(SubtitlePairWords(words_in_captions, caption, similarity))

    return pairs


def check_forward(i: int, subtitle_pairs: List[SubtitlePairWords], total_diff: int):
    this_pair = copy.deepcopy(subtitle_pairs[i])

    next_one = copy.deepcopy(subtitle_pairs[i + 1])

    this_pair.words.append(next_one.words[0])
    next_one.words.remove(next_one.words[0])

    this_pair.similarity = get_ratio(this_pair.caption, this_pair.words)[1]
    next_one.similarity = get_ratio(next_one.caption, next_one.words)[1]

    new_windowed_array = [subtitle_pairs[i - 1], this_pair, next_one]
    new_diff = sum(x.similarity for x in new_windowed_array)
    # print(f'new diff {new_diff}')

    if new_diff > total_diff:
        best_one = copy.deepcopy(this_pair)
        best_one_next = copy.deepcopy(next_one)
        diff_so_far = total_diff

        while new_diff > diff_so_far:
            best_one = copy.deepcopy(this_pair)
            best_one_next = copy.deepcopy(next_one)

            diff_so_far = new_diff

            if len(next_one.words) == 0:
                break

            this_pair.words.append(next_one.words[0])
            next_one.words.remove(next_one.words[0])

            this_pair.similarity = get_ratio(this_pair.caption, this_pair.words)[1]
            next_one.similarity = get_ratio(next_one.caption, next_one.words)[1]

            new_windowed_array = [subtitle_pairs[i - 1], this_pair, next_one]
            new_diff = sum(x.similarity for x in new_windowed_array)
        return best_one, best_one_next, diff_so_far
    return subtitle_pairs[i], subtitle_pairs[i + 1], total_diff


def check_backwards(i: int, subtitle_pairs: List[SubtitlePairWords], total_diff: int):
    this_pair = copy.deepcopy(subtitle_pairs[i])

    next_one = copy.deepcopy(subtitle_pairs[i + 1])

    next_one.words.insert(0, this_pair.words[-1])
    this_pair.words.remove(this_pair.words[-1])

    this_pair.similarity = get_ratio(this_pair.caption, this_pair.words)[1]
    next_one.similarity = get_ratio(next_one.caption, next_one.words)[1]

    new_windowed_array = [subtitle_pairs[i - 1], this_pair, next_one]
    new_diff = sum(x.similarity for x in new_windowed_array)
    # print(f'new diff {new_diff}')

    if new_diff > total_diff:
        best_one = copy.deepcopy(this_pair)
        best_one_next = copy.deepcopy(next_one)
        diff_so_far = total_diff

        while new_diff > diff_so_far:
            best_one = copy.deepcopy(this_pair)
            best_one_next = copy.deepcopy(next_one)

            diff_so_far = new_diff

            if len(this_pair.words) == 0:
                break

            next_one.words.insert(0, this_pair.words[-1])
            this_pair.words.remove(this_pair.words[-1])

            this_pair.similarity = get_ratio(this_pair.caption, this_pair.words)[1]
            next_one.similarity = get_ratio(next_one.caption, next_one.words)[1]

            new_windowed_array = [subtitle_pairs[i - 1], this_pair, next_one]
            new_diff = sum(x.similarity for x in new_windowed_array)
        return best_one, best_one_next, diff_so_far
    return subtitle_pairs[i], subtitle_pairs[i + 1], total_diff


def moving_window(file_name):
    pairs = process_subtitles(file_name, True)
    window_size = 3
    for i, pair in enumerate(pairs):

        if i <= (window_size // 2):
            start = i
            end = i + (window_size // 2) + 1
        elif i + (window_size // 2) >= len(pairs):
            start = i - window_size
            end = i
        else:
            start = i - (window_size // 2)
            end = i + (window_size // 2) + 1

        windowed_array = pairs[start:end]

        total_diff = sum(x.similarity for x in windowed_array)
        # print(f'old diff {total_diff}')

        if pair.similarity < 2:
            continue

        if i == len(pairs) - 1:
            break

        if len(pair.words) == 0 or len(pairs[i + 1].words) == 0:
            continue

        if i == 0:
            # print(pairs[0])
            forward_this_pair = copy.deepcopy(pair)

            forward_next_one = copy.deepcopy(pairs[i + 1])

            forward_this_pair.words.append(forward_next_one.words[0])
            forward_next_one.words.remove(forward_next_one.words[0])

            forward_this_pair.similarity = get_ratio(forward_this_pair.caption, forward_this_pair.words)[1]
            forward_next_one.similarity = get_ratio(forward_next_one.caption, forward_next_one.words)[1]

            forward_new_windowed_array = [forward_this_pair, forward_next_one]
            forward_new_diff = sum(x.similarity for x in forward_new_windowed_array)

            backwards_this_pair = copy.deepcopy(pair)

            backwards_next_one = copy.deepcopy(pairs[i + 1])

            backwards_next_one.words.insert(0, backwards_this_pair.words[-1])
            backwards_this_pair.words.remove(backwards_this_pair.words[-1])

            backwards_this_pair.similarity = get_ratio(backwards_this_pair.caption, backwards_this_pair.words)[1]
            backwards_next_one.similarity = get_ratio(backwards_next_one.caption, backwards_next_one.words)[1]

            backwards_new_windowed_array = [backwards_this_pair, backwards_next_one]
            backwards_new_diff = sum(x.similarity for x in backwards_new_windowed_array)

            if backwards_new_diff > total_diff and backwards_new_diff > forward_new_diff:
                pairs[i] = backwards_this_pair
                pairs[i + 1] = backwards_next_one

            if forward_new_diff > total_diff and forward_new_diff > backwards_new_diff:
                pairs[i] = forward_this_pair
                pairs[i + 1] = forward_next_one

        else:
            # print(pair)
            forward_this_pair, forward_next_one, forward_new_diff = check_forward(i, pairs, total_diff)
            backwards_this_pair, backwards_next_one, backwards_new_diff = check_backwards(i, pairs, total_diff)

            if backwards_new_diff > total_diff and backwards_new_diff > forward_new_diff:
                pairs[i] = backwards_this_pair
                pairs[i + 1] = backwards_next_one

            if forward_new_diff > total_diff and forward_new_diff > backwards_new_diff:
                pairs[i] = forward_this_pair
                pairs[i + 1] = forward_next_one

    return pairs


if __name__ == '__main__':
    # files_no_ext = set([os.path.splitext(f)[0] for f in os.listdir('data/')])
    # number_of_files = len(files_no_ext)
    # for i, file_name in enumerate(files_no_ext):
    #     print(f"Processing {file_name} {i + 1} out of {number_of_files}")
    #     if file_name.startswith("."):
    #         continue
    #     if not os.path.exists(f'data/{file_name}.json') or not os.path.exists(f'data/{file_name}.vtt'):
    #         print(f"no file pair for {file_name}")
    #         continue
    #     try:
    #         pairs = moving_window(file_name)
    #         with open(f'training-data/{file_name}.json', 'w', encoding='utf-8') as f:
    #             json.dump(pairs, f, ensure_ascii=False, indent=4, cls=SubtitlePairWordsEncoder)
    #     except:
    #         print(f"Error with {file_name}")

    name = 'foorum-370'
    pairs = moving_window(name)
    result_file_name = f'fuzziness/{name}.txt'

    with open(f'fuzziness/{name}-no-sort.json', 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=4, cls=SubtitlePairWordsEncoder)

    #
    # with open(result_file_name, encoding='utf-8', errors='ignore', mode="w+") as result_file:
    #     for pair in pairs:
    #         result_file.write(pair.__str__())
    #         result_file.write("\n\n")
        # mean = np.mean([item.similarity for item in pairs])
        # result_file.write(f"Mean difference: {mean}")
