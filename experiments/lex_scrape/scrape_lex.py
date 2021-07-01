#!/usr/bin/env python3

import pickle
import os
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from pyyoutube import Api, PlaylistItem
from typing import List
from config import API_KEY

api = Api(api_key=API_KEY)


def extract_text(transcript):
    res = ""
    for item in transcript:
        res += item["text"] + " "
    return res


playlists = {
    "podcasts": "PLrAXtmErZgOdP_8GztsuKi9nrraNbKKp4",
    #'misc': 'PLrAXtmErZgOcl7mvyfkQTHFnOGZxWtN55',
    #'ama': 'PLrAXtmErZgOdEfD2VtObCncE4psXYAcpq',
    #'life': 'PLrAXtmErZgOe9yLXwWFRO4UMHN1s3GXIk',
}

if __name__ == "__main__":

    texts = {}
    number = 1

    for playlist_id in playlists.values():

        items: List[PlaylistItem] = api.get_playlist_items(
            playlist_id=playlist_id, count=None
        ).items

        print(f"Processing {len(items)} videos in playlist {playlist_id}")

        for item in items:
            video_id = item.snippet.resourceId.videoId
            print(f"Processing video #{number} https://youtube.com/watch?v={video_id}")
            number += 1
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
            except (TranscriptsDisabled, NoTranscriptFound):
                continue
            res = extract_text(transcript)
            l = min(len(res), 80)
            print(res[:l] + " ...")
            texts[video_id] = res

    if not os.path.isdir("data"):
        os.mkdir("data")

    with open("data/transcripts.pkl", "wb") as stream:
        pickle.dump(texts, stream)
