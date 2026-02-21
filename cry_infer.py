import numpy as np
import librosa

def audio_to_features(audio_1d, sr, target_sr=22050, n_mfcc=20):

    # Resample if needed
    if sr != target_sr:
        audio_1d = librosa.resample(
            audio_1d.astype(np.float32),
            orig_sr=sr,
            target_sr=target_sr
        )
        sr = target_sr

    # Ensure 1 second audio
    min_len = sr * 1
    if len(audio_1d) < min_len:
        audio_1d = np.pad(audio_1d, (0, min_len - len(audio_1d)))
    else:
        audio_1d = audio_1d[:min_len]

    mfcc = librosa.feature.mfcc(y=audio_1d, sr=sr, n_mfcc=n_mfcc)

    feat_mean = np.mean(mfcc, axis=1)
    feat_std = np.std(mfcc, axis=1)

    features = np.concatenate([feat_mean, feat_std], axis=0)

    return features.reshape(1, -1).astype(np.float32)