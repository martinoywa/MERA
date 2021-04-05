import requests
import json


API_URL = "https://api.musixmatch.com/ws/1.1/"
SECRET_KEY = "37d7829ae9c5bf71588c67bccf16cdde"


def lyrics_finder(out_format, callback, track, artist, apikey):
    """
    Queries the MusixMatch Lyrics finder API
    :param out_format: json, jsonp, xml
    :param callback: jsonp, callback
    :param track: The song title
    :param artist: The song artist
    :param apikey: Developer's API Key
    :return: status code and lyrics
    """

    response = requests.get(
        f"{API_URL}matcher.lyrics.get?format={out_format}&callback={callback}&"
        f"q_track={track}&q_artist={artist}&apikey={apikey}")
    content = json.loads(response.content.decode("utf-8")[9:-2])
    status = content["message"]["header"]["status_code"]
    lyrics = content["message"]["body"]

    return status, lyrics


# # EXAMPLE:
# if __name__ == "__main__":
#     status_code, lyrics = lyrics_finder("jsonp", "callback", "Bleeding Love", "Leona Lewis",
#                                         SECRET_KEY)
#     # print(status_code)
#     print(lyrics)
