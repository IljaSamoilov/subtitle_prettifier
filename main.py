import json
from typing import List

import webvtt

from sections import SubtitlePair, TranscriptSection

subtiltes = webvtt.read('data/foorum-4-aasta-eelarvestrateegia.vtt')

with open('data/foorum-4-aasta-eelarvestrateegia.json', encoding='utf-8', errors='ignore') as fh:
    json_text = json.load(fh)


def is_caption_in_transcript(caption: webvtt.Caption, transcript_dict: dict) -> bool:
    return transcript_dict['end'] > caption.end_in_seconds


generated_subtitles = json_text['sections'][0]['turns']

i = 0
pairs: List[SubtitlePair] = []

for transcript in generated_subtitles:

    captions_in_transcript: List[webvtt.Caption] = []

    while i < len(subtiltes.captions) and is_caption_in_transcript(subtiltes.captions[i], transcript):
        captions_in_transcript.append(subtiltes.captions[i])
        i += 1

    pairs.append(SubtitlePair(TranscriptSection(transcript), captions_in_transcript))

for pair in pairs:
    print(pair)
    print(f"Cosine similarity: {pair.get_cosine_confidence()}")
    print("\n")
