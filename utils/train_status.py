import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# === Load dataset ===
df = pd.read_csv("data/voice_dataset.csv")

# === Fitur 36 kolom ===
feature_cols = [col for col in df.columns if col.startswith("mfcc")]
X = df[feature_cols].to_numpy()
y = df["status"].to_numpy()  # 0=buka, 1=tutup

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Pipeline RandomForest + StandardScaler ===
model_status = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])

# === Training ===
model_status.fit(X_train, y_train)

# === Simpan model dan feature order ===
os.makedirs("models", exist_ok=True)
joblib.dump(model_status, "models/status_model.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")
print("[INFO] Model status berhasil disimpan.")
