import json
from typing import List, Dict, Tuple
import os

import webvtt
import numpy as np

from sections import SubtitlePair, TranscriptSection, SubtitlePairWords, SubtitlePairWordsEncoder

from utility import is_word_in_caption, get_similarity

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

        cos_similarity, lev_distance = get_similarity(caption, words_in_captions)
        similarity = lev_distance if use_levishtein else cos_similarity

        pairs.append(SubtitlePairWords(words_in_captions, caption, similarity))

    return pairs


def moving_window(file_name):
    pairs = process_subtitles(file_name, True)
    window_size = 3
    for i, pair in enumerate(pairs):

        if i <= (window_size // 2):
            start = i
            end = i + window_size
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
            this_pair = copy.deepcopy(pair)

            next_one = copy.deepcopy(pairs[i + 1])

            this_pair.words.append(next_one.words[0])
            next_one.words.remove(next_one.words[0])

            this_pair.similarity = get_similarity(this_pair.caption, this_pair.words)[1]
            next_one.similarity = get_similarity(next_one.caption, next_one.words)[1]

            new_windowed_array = [this_pair, next_one, pairs[i + 2]]
            new_diff = sum(x.similarity for x in new_windowed_array)
            # print(f'new diff {new_diff}')

            if new_diff < total_diff:
                pairs[i] = this_pair
                pairs[i + 1] = next_one
        else:
            # print(pair)

            this_pair = copy.deepcopy(pair)

            next_one = copy.deepcopy(pairs[i + 1])

            this_pair.words.append(next_one.words[0])
            next_one.words.remove(next_one.words[0])

            this_pair.similarity = get_similarity(this_pair.caption, this_pair.words)[1]
            next_one.similarity = get_similarity(next_one.caption, next_one.words)[1]

            new_windowed_array = [pairs[i - 1], this_pair, next_one]
            new_diff = sum(x.similarity for x in new_windowed_array)
            # print(f'new diff {new_diff}')

            if new_diff < total_diff:

                while new_diff < total_diff:
                    best_one = copy.deepcopy(this_pair)
                    best_one_next = copy.deepcopy(next_one)

                    total_diff = new_diff

                    if len(next_one.words) == 0:
                        break

                    this_pair.words.append(next_one.words[0])
                    next_one.words.remove(next_one.words[0])

                    this_pair.similarity = get_similarity(this_pair.caption, this_pair.words)[1]
                    next_one.similarity = get_similarity(next_one.caption, next_one.words)[1]

                    new_windowed_array = [pairs[i - 1], this_pair, next_one]
                    new_diff = sum(x.similarity for x in new_windowed_array)

                pairs[i] = best_one
                pairs[i + 1] = best_one_next
    return pairs


if __name__ == '__main__':
    files_no_ext = set([os.path.splitext(f)[0] for f in os.listdir('data/')])

    for file_name in files_no_ext:
        if file_name.startswith("."):
            continue
        if not os.path.exists(f'data/{file_name}.json') or not os.path.exists(f'data/{file_name}.vtt'):
            print(f"no file pair for {file_name}")
            continue
        try:
            pairs = moving_window(file_name)
            with open(f'training-data/{file_name}.json', 'w', encoding='utf-8') as f:
                json.dump(pairs, f, ensure_ascii=False, indent=4, cls=SubtitlePairWordsEncoder)
        except:
            print(f"Error with {file_name}")
    # result_file_name = f'results/{name}-window-6-not-sorted.txt'

    # with open(result_file_name, encoding='utf-8', errors='ignore', mode="a+") as result_file:
    #     for pair in pairs:
    #         result_file.write(pair.__str__())
    #         result_file.write("\n\n")
    #     mean = np.mean([item.similarity for item in pairs])
    #     result_file.write(f"Mean similarity: {mean}")
