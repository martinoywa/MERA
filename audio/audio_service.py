import os
from youtube_search import YoutubeSearch
import youtube_dl
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp


# configuration options
ydl_opts = {
    'format': 'worstaudio/worst',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'postprocessor_args': [
        '-ac', '1'  # mono channel
    ],
}


def create_spectrogram(id, track, artist):
    """
    Finds songs on YouTube using track and artist name then
    saves the song's spectrogram.
    :param id: Deezer Song ID
    :param track: The song name
    :param artist: The song artist
    :return: song spectrogram
    """

    search_name = track + " by " + artist
    res = YoutubeSearch(search_name, max_results=1).to_dict()
    link = "https://www.youtube.com" + res[0]["url_suffix"]

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

    source = os.listdir()
    for file in source:
        if file.endswith(".mp3"):
            print(file)
            mp3_audio = AudioSegment.from_file(file, format="mp3")  # read mp3
            wname = mktemp('.wav')  # use temporary file
            mp3_audio.export(wname, format="wav")  # convert to wav
            FS, data = wavfile.read(wname)  # read wav file

            # plot spectrogram
            plt.specgram(data, Fs=FS, NFFT=1024, noverlap=0)
            plt.xticks([])
            plt.yticks([])

            # save the spectrogram
            plt.savefig(f"spectrograms/{id}", dpi=300)
            plt.close()
            os.remove(file)
            break
