import json
from typing import List, Dict, Tuple

import webvtt

from sections import SubtitlePair, TranscriptSection, SubtitlePairWords

import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


subtiltes = webvtt.read('data/foorum-4-aasta-eelarvestrateegia.vtt')

with open('data/foorum-4-aasta-eelarvestrateegia.json', encoding='utf-8', errors='ignore') as fh:
    json_text = json.load(fh)


def clean_string(text):
    text = text.replace('\n', " ")
    return ''.join([word for word in text if word not in string.punctuation]).lower()

def is_word_in_caption(capt: webvtt.Caption, word_dict: dict) -> bool:
    return capt.end_in_seconds > word_dict['end']

def get_cosine_similarity(this_caption: webvtt.Caption, word_dict: List[Dict]) -> float:

    real = clean_string(this_caption.text)
    generated = ' '.join([x['word'] for x in word_dict])
    generated = clean_string(generated)

    arr = [real, generated]

    vectorizer = CountVectorizer().fit_transform(arr)
    vectors = vectorizer.toarray()

    vec1 = vectors[0].reshape(1, -1)
    vec2 = vectors[1].reshape(1, -1)

    return cosine_similarity(vec1, vec2)[0][0]


generated_subtitles = json_text['sections'][0]['turns']

i = 0
pairs: List[SubtitlePairWords] = []

words_arrays = list(map(lambda x: x['words'], generated_subtitles))
words = [item for sublist in words_arrays for item in sublist]

for caption in subtiltes.captions:

    words_in_captions: List[Dict] = []

    while i < len(words) and is_word_in_caption(caption, words[i]):
        words_in_captions.append(words[i])
        i += 1

    similarity: float = get_cosine_similarity(caption, words_in_captions)

    if i < len(words):
        with_next_word = words_in_captions + [words[i]]
        one_less_word = words_in_captions[:-1]

        similarity_with_next_word = get_cosine_similarity(caption, with_next_word)
        similarity_one_less_word = get_cosine_similarity(caption, one_less_word)

        if similarity_with_next_word > similarity or similarity_one_less_word > similarity:
            if similarity_with_next_word > similarity_one_less_word:
                i += 1
                words_in_captions = with_next_word
                similarity = similarity_with_next_word
            else:
                i -= 1
                words_in_captions = one_less_word
                similarity = similarity_one_less_word

    pairs.append(SubtitlePairWords(words_in_captions, caption, similarity))

with open('word-result.txt', encoding='utf-8', errors='ignore', mode="a+") as result_file:
    for pair in pairs:
        result_file.write(pair.__str__())
        result_file.write("\n\n")
