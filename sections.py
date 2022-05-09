from typing import List, Dict
from webvtt import Caption
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from json import JSONEncoder


def clean_string(text):
    return ''.join([word for word in text if word not in string.punctuation]).lower()


class TranscriptSection:

    def __init__(self, section_dict: dict):
        self.speaker = section_dict['speaker']
        self.start = section_dict['start']
        self.end = section_dict['end']
        self.transcript = section_dict['transcript']


class SubtitlePair:

    def __init__(self, transcript: TranscriptSection, sections: List[Caption]):
        self.transcript = transcript
        self.sections = sections

    def get_texts(self) -> (str, str):
        subtitle_text = ' '.join([x.text.replace('\n', " ") for x in self.sections])
        return self.transcript.transcript, subtitle_text

    # https://towardsdatascience.com/calculating-string-similarity-in-python-276e18a7d33a
    def get_cosine_confidence(self) -> float:
        texts = self.get_texts()

        generated = clean_string(texts[0])
        real_one = clean_string(texts[1])

        arr = [generated, real_one]

        vectorizer = CountVectorizer().fit_transform(arr)
        vectors = vectorizer.toarray()

        vec1 = vectors[0].reshape(1, -1)
        vec2 = vectors[1].reshape(1, -1)

        return cosine_similarity(vec1, vec2)[0][0]

    def __str__(self):
        texts = self.get_texts()
        return f"Generated: {texts[0]} \nSubtitle: {texts[1]}"

    def __repr__(self):
        return self.__str__()


class SubtitlePairWords:

    def __init__(self, words: List[Dict], caption: Caption, similarity: float):
        self.words = words
        self.caption = caption
        self.similarity = similarity

    def get_texts(self) -> (str, str):
        subtitle_text = ' '.join([x['word_with_punctuation'] for x in self.words])
        return subtitle_text.replace('\n', " "), self.caption.text.replace('\n', " ")

    def __str__(self):
        texts = self.get_texts()
        return f"Generated: {texts[0]} \nSubtitle: {texts[1]} \nDistance: {self.similarity}"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        texts = self.get_texts()
        return {
            "distance": self.similarity,
            "generated": texts[0],
            "subtitle": texts[1],
            "seconds": self.caption.end_in_seconds - self.caption.start_in_seconds
        }


class SubtitlePairWordsEncoder(JSONEncoder):
    def default(self, o):
        return o.to_dict()
