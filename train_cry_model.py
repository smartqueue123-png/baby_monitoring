import os
import numpy as np
import pandas as pd
import joblib

import librosa

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

ESC_ROOT = "ESC-50-master"
CSV_PATH = os.path.join(ESC_ROOT, "meta", "esc50.csv")
AUDIO_DIR = os.path.join(ESC_ROOT, "audio")

OUT_MODEL = "cry_model.pkl"

# Feature extraction: MFCC mean + std
def extract_mfcc_features(wav_path, target_sr=22050, n_mfcc=20):
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    # Ensure fixed minimum length (~1 sec) by padding
    min_len = target_sr * 1
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat_mean = np.mean(mfcc, axis=1)
    feat_std = np.std(mfcc, axis=1)
    feats = np.concatenate([feat_mean, feat_std], axis=0)
    return feats.astype(np.float32)

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("Cannot find esc50.csv at: " + CSV_PATH)
    if not os.path.exists(AUDIO_DIR):
        raise FileNotFoundError("Cannot find audio folder at: " + AUDIO_DIR)

    df = pd.read_csv(CSV_PATH)

    # Positive class: crying_baby
    df["label"] = (df["category"] == "crying_baby").astype(int)

    # Make dataset balanced so it trains well and doesn't bias toward "not cry"
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0].sample(n=len(pos), random_state=42)

    use_df = pd.concat([pos, neg], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    X_list = []
    y_list = []

    print("Extracting MFCC features...")
    for i, row in use_df.iterrows():
        fname = row["filename"]
        wav_path = os.path.join(AUDIO_DIR, fname)

        if not os.path.exists(wav_path):
            # skip missing
            continue

        feats = extract_mfcc_features(wav_path)
        X_list.append(feats)
        y_list.append(int(row["label"]))

        if (i + 1) % 50 == 0:
            print("Processed", i + 1, "files")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int32)

    print("Total samples:", len(y), " Cry:", int(np.sum(y)), " NotCry:", int(len(y) - np.sum(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000))
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["NOT_CRY", "CRY"]))

    joblib.dump(model, OUT_MODEL)
    print("\nSaved model to:", OUT_MODEL)

if __name__ == "__main__":
    main()