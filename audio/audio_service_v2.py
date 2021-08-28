import os
from youtube_search import YoutubeSearch
import youtube_dl
import torch
import torchaudio

import librosa
import matplotlib.pyplot as plt

import numpy as np

import shutil


# configuration options
ydl_opts = {
    'format': 'bestaudio/best',
    'max-filesize': '20M',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}

# parameters
SAMPLE_RATE = 22050
NFFT = 2048
HL = 512
MELS = 128
NUMBER_OF_SAMPLES = 200000

# mel spectrogram transformer
transformer = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=NFFT,
    hop_length=HL,
    n_mels=MELS,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm='slaney',
    onesided=True,
    mel_scale="htk"
)

# mix down or convert to mono channel
def mix_down_if_necessary(signal):
    # stereo [2, 128, 1024]
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

# resample to SAMPLE_RATE
def resample_if_necessary(signal, sr):
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=SAMPLE_RATE
        )
        signal = resampler(signal)
    return signal

# cut signal if longer than NUMBER_OF_SAMPLES
def cut_if_necessary(signal):
    if signal.shape[1] > NUMBER_OF_SAMPLES:
        signal = signal[:, :NUMBER_OF_SAMPLES]
    return signal

# right pad if necessary if less than NUMBER_OF_SAMPLES
def right_pad_if_necessary(signal):
    if signal.shape[1] < NUMBER_OF_SAMPLES:
        num_to_pad = NUMBER_OF_SAMPLES - signal.shape[1]
        signal = torch.nn.functional.pad(signal, (0,num_to_pad))
    return signal


def create_mel_spectrogram(id, track, artist, path):
    """
    Finds songs on YouTube using track and artist name then
    saves the song's mel spectrogram.
    :param id: Deezer Song ID
    :param track: The song name
    :param artist: The song artist
    :param path: path to store downloaded spectrograms
    :return: song spectrogram
    """

    search_name = track + " by " + artist
    res = YoutubeSearch(search_name, max_results=1).to_dict()

    # only download if both track and artist name in res title
    if track in res[0]["title"] and artist in res[0]["title"]:
        link = "https://www.youtube.com" + res[0]["url_suffix"]

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])

        source = os.listdir()
        # rename file to match song id
        for file in source:
            if file.endswith(res[0]["id"] + ".mp3"):
                os.rename(file, str(id)+".mp3")

        # find song by id and save mel spectrogram
        file = str(id)+".mp3"
        signal, sr = torchaudio.load(file)

        signal = mix_down_if_necessary(signal)
        signal = resample_if_necessary(signal, sr)
        signal = cut_if_necessary(signal)
        signal = right_pad_if_necessary(signal)
        signal = transformer(signal)

        # save the mel spectrogram
        plt.imsave(f"{path}/{id}.png", librosa.power_to_db(signal[0].numpy(), ref=np.max), origin="lower", format='png')
        plt.close()

        try:
            shutil.move(file, "/home/martinoywa/Music/Project/")
        except:
            pass
    else:
        pass
