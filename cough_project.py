import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---- 1. SET YOUR EXACT PATHS HERE ----
healthy_path = r"C:\Users\tabas\OneDrive\Desktop\Cough_Project\dataset\healthy"
unhealthy_path = r"C:\Users\tabas\OneDrive\Desktop\Cough_Project\dataset\unhealthy"

# ---- 2. FEATURE EXTRACTION ----
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfccs

# ---- 3. LOAD DATA ----
data = []
labels = []

# Healthy files
for file in os.listdir(healthy_path):
    if file.endswith(".wav"):
        file_path = os.path.join(healthy_path, file)
        print("Reading:", file_path)
        try:
            features = extract_features(file_path)
            data.append(features)
            labels.append("healthy")
        except Exception as e:
            print("Error reading", file, ":", e)

# Unhealthy files
for file in os.listdir(unhealthy_path):
    if file.endswith(".wav"):
        file_path = os.path.join(unhealthy_path, file)
        print("Reading:", file_path)
        try:
            features = extract_features(file_path)
            data.append(features)
            labels.append("unhealthy")
        except Exception as e:
            print("Error reading", file, ":", e)

# ---- 4. CHECK DATA ----
print("Total files loaded:", len(data))

if len(data) < 2:
    print("ERROR: Not enough valid audio files. Add at least 1 .wav in each folder.")
    exit()

# ---- 5. TRAIN MODEL ----
X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---- 6. TEST MODEL ----
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---- 7. USER TEST ----
test_file = input("Enter test audio file path: ")
features = extract_features(test_file).reshape(1, -1)
prediction = model.predict(features)

print("Prediction:", prediction[0])