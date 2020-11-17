import json
from typing import List, Dict, Tuple

import webvtt
import difflib
from nltk import edit_distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint

file_name = "suud-puhtaks-15"

subtitles = webvtt.read(f'data/{file_name}.vtt')

with open(f'data/{file_name}.json', encoding='utf-8', errors='ignore') as fh:
    json_text = json.load(fh)

generated_subtitles = [x['turns'] for x in json_text['sections'] if (x['type'] == "speech" and "turns" in x.keys())]
generated_subtitles = [item for sublist in generated_subtitles for item in sublist]

generated_text = "\n".join([x['transcript'] for x in generated_subtitles]).lower()

subtitles_text = "\n".join([x.text.replace('\n', " ") for x in subtitles.captions]).lower()

seq = difflib.SequenceMatcher(a=generated_text, b=subtitles_text, autojunk=True)
print(seq.ratio())


# str1 = "millest see konkreetne jutt praegu"
str1 = "Me ei tea neid summasid meid ja neid isikuid"
# str2 = "millest konkreetne jutt käib"
str2 = "Me ei tea neid summasid, me ei tea neid isikuid"


def get_similarities(str1, str2):
    seq2 = difflib.SequenceMatcher(a=str1, b=str2)
    print(f"Difflib:{seq2.ratio()}")
    # levenshtein_distance = edit_distance(generated_text, subtitles_text)
    arr = [str1, str2]
    vectorizer = CountVectorizer().fit_transform(arr)
    vectors = vectorizer.toarray()
    vec1 = vectors[0].reshape(1, -1)
    vec2 = vectors[1].reshape(1, -1)
    cosine = cosine_similarity(vec1, vec2)[0][0]
    print(f"Cosine:{cosine}")
    print(f"Levenshtein {edit_distance(str1, str2)}")


d = difflib.Differ()
result = list(d.compare(str1, str2))
pprint(result)

get_similarities(str1, str2)