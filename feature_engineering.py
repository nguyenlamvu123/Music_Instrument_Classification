import librosa
import numpy as np
from tqdm import tqdm
import sklearn
from config import CreateDataset, sr

sr = CreateDataset.sr
fs = CreateDataset.fs
hs = CreateDataset.hs
mfcc_dim = CreateDataset.mfcc_dim
cs = CreateDataset.cs
ms = CreateDataset.ms
ts = CreateDataset.ts
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))


def readdata(piece):
    is_created = False
    dataset_numpy = None
    for audio in tqdm(piece):
        samples, _ = librosa.load(audio, sr=sr, duration=4.0)
        row = extract_feature(samples)
        if not is_created:
            dataset_numpy = np.array(row)
            is_created = True
        elif is_created:
            dataset_numpy = np.vstack((dataset_numpy, row))
    assert dataset_numpy is not None
    return scaler.fit_transform(dataset_numpy)

def extract_feature(samples):
    result = []
    features = []

    # Timbre features
    spectral_centroid = librosa.feature.spectral_centroid(y=samples, sr=sr, n_fft=fs, hop_length=hs)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=samples, sr=sr, n_fft=fs, hop_length=hs)
    spectral_contrast = librosa.feature.spectral_contrast(y=samples, sr=sr, n_fft=fs, hop_length=hs)
    spectral_rollof = librosa.feature.spectral_rolloff(y=samples, sr=sr, n_fft=fs, hop_length=hs)
    spectral_flux = librosa.onset.onset_strength(y=samples, sr=sr, center=True)
    zero_crossing = librosa.feature.zero_crossing_rate(y=samples, frame_length=fs, hop_length=hs)
    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_fft=fs, hop_length=hs)
    mel_scale = librosa.feature.melspectrogram(y=samples, n_fft=fs, hop_length=hs, sr=sr)
    mel_scale = librosa.power_to_db(mel_scale)
    # Rhythm features
    tempo = librosa.beat.tempo(y=samples, sr=sr, hop_length=hs)

    # Pitch/Harmony features
    # chroma = librosa.feature.chroma_stft(y=samples, sr=sr, hop_length=hs, n_fft=fs, n_chroma=cs)
    # tonal_centroid = librosa.feature.tonnetz(y=samples, sr=sr)

    # add
    features.append(spectral_contrast)
    features.append(spectral_bandwidth)
    features.append(spectral_centroid)
    features.append(spectral_rollof)
    features.append(spectral_flux)
    features.append(zero_crossing)
    features.append(tempo)

    for feature in features:
        result.append(np.mean(feature))
        result.append(np.std(feature))

    for i in range(0, mfcc_dim):
        result.append(np.mean(mfcc[i,:]))
        result.append(np.std(mfcc[i, :]))

    # for i in range(0, cs):
    #     result.append(np.mean(chroma[i, :]))
    #     result.append(np.std(chroma[i, :]))

    for i in range(0, ms):
        result.append(np.mean(mel_scale[i, :]))
        result.append(np.std(mel_scale[i, :]))

    # for i in range(0, ts):
    #     result.append(np.mean(tonal_centroid[i, :]))
    #     result.append(np.std(tonal_centroid[i, :]))

    return result
