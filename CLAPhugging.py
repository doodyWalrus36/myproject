import os
import ssl
import certifi
import yt_dlp as youtube_dl
from pydub import AudioSegment
import numpy as np
import soundfile as sf
from transformers import ClapModel, ClapProcessor
from scipy.spatial.distance import cosine
import pandas as pd
import torch
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Set SSL_CERT_FILE environment variable
os.environ['SSL_CERT_FILE'] = certifi.where()

# Ensure SSL certificates are used
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize the CLAP model and processor
model_name = "laion/clap-htsat-unfused"
processor = ClapProcessor.from_pretrained(model_name)
model = ClapModel.from_pretrained(model_name)

def download_audio(url, output_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path + '.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
            'nopostoverwrites': False,
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def preprocess_audio(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(48000) 
    audio.export(output_path, format='wav')

def extract_audio_embedding(audio_path):
    audio, sr = sf.read(audio_path)
    inputs = processor(audios=audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_audio_features(**inputs).squeeze().numpy()
    return embedding

def compute_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1.flatten(), embedding2.flatten())

# Load dataset from CSV with specified encoding
data_path = '/Users/army/Desktop/datainCSV.csv'
data = pd.read_csv(data_path, encoding='ISO-8859-1').to_dict(orient='records')

# Define a path to save and load embeddings
embeddings_file = '/Users/army/Desktop/song_embeddings.json'

# Load existing embeddings if the file exists
if os.path.exists(embeddings_file):
    with open(embeddings_file, 'r') as f:
        embeddings_dict = json.load(f)
else:
    embeddings_dict = {}

results_yes = []
results_no = []
results_control = []

for entry in data:
    original_work = entry['Original Work']
    second_song = entry['Second Song']
    court_decision = entry['Court decision']

    # Check if the embeddings already exist
    if original_work in embeddings_dict:
        embedding_original = np.array(embeddings_dict[original_work])
    else:
        original_audio_path = f"audio_{original_work.replace(' ', '_')}"
        processed_audio_path = f"processed_audio_{original_work.replace(' ', '_')}.wav"

        download_audio(entry['First Song Link'], original_audio_path)
        preprocess_audio(original_audio_path + '.wav', processed_audio_path)
        
        embedding_original = extract_audio_embedding(processed_audio_path)
        embeddings_dict[original_work] = embedding_original.tolist()
        print(f"Original embedding for {original_work}: {embedding_original}")

    if second_song in embeddings_dict:
        embedding_second = np.array(embeddings_dict[second_song])
    else:
        second_audio_path = f"audio_{second_song.replace(' ', '_')}"
        processed_second_audio_path = f"processed_audio_{second_song.replace(' ', '_')}.wav"

        download_audio(entry['Second Song Link'], second_audio_path)
        preprocess_audio(second_audio_path + '.wav', processed_second_audio_path)
        
        embedding_second = extract_audio_embedding(processed_second_audio_path)
        embeddings_dict[second_song] = embedding_second.tolist()
        print(f"Second embedding for {second_song}: {embedding_second}")

    similarity = compute_similarity(embedding_original, embedding_second)
    print(f"Similarity between {original_work} and {second_song}: {similarity}")
    
    result = {
        "Original Work": original_work,
        "Second Song": second_song,
        "Similarity": similarity,
        "Court decision": court_decision
    }

    if court_decision == "Yes":
        results_yes.append(result)
    elif court_decision == "No":
        results_no.append(result)
    else:
        results_control.append(result)

# Save embeddings to a file for future use
with open(embeddings_file, 'w') as f:
    json.dump(embeddings_dict, f)

# Create DataFrames
df_results_yes = pd.DataFrame(results_yes)
df_results_no = pd.DataFrame(results_no)
df_results_control = pd.DataFrame(results_control)

# Visualization: Cosine Similarity vs Court Decision
plt.figure(figsize=(12, 6))

# Plotting results where court decision was "Yes"
plt.scatter([result['Original Work'] for result in results_yes], 
            [result['Similarity'] for result in results_yes], 
            color='green', label='Yes')

# Plotting results where court decision was "No"
plt.scatter([result['Original Work'] for result in results_no], 
            [result['Similarity'] for result in results_no], 
            color='red', label='No')

# Plotting results where court decision is the controlled group
plt.scatter([result['Original Work'] for result in results_control], 
            [result['Similarity'] for result in results_control], 
            color='blue', label='Control')

# Customize the plot
plt.xticks(rotation=90)
plt.xlabel("Original Work")
plt.ylabel("Cosine Similarity")
plt.title("Cosine Similarity vs Court Decision")
plt.legend(loc='upper right')
plt.tight_layout()

# Show the plot
plt.show()

# Prepare data for machine learning
df_all = pd.concat([df_results_yes, df_results_no, df_results_control])
X = df_all[['Similarity']]  # Use similarity and potentially other features
y = df_all['Court decision']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into train and test sets before oversampling
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Apply SMOTE only to the training data
smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=2)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize models with class_weight adjustment
svm_model = SVC(kernel='linear', random_state=42, class_weight='balanced')
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
gb_model = GradientBoostingClassifier(random_state=42)
dummy_model = DummyClassifier(strategy='stratified', random_state=42)

# Hyperparameter tuning with Grid Search
param_grid_svm = {'C': [0.1, 1, 10]}
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
param_grid_dt = {'max_depth': [None, 10, 20]}

svm_grid = GridSearchCV(svm_model, param_grid_svm, cv=5)
rf_grid = GridSearchCV(rf_model, param_grid_rf, cv=5)
dt_grid = GridSearchCV(dt_model, param_grid_dt, cv=5)

# Train models with the best parameters
svm_grid.fit(X_train_resampled, y_train_resampled)
rf_grid.fit(X_train_resampled, y_train_resampled)
dt_grid.fit(X_train_resampled, y_train_resampled)
gb_model.fit(X_train_resampled, y_train_resampled)
dummy_model.fit(X_train_resampled, y_train_resampled)

# Evaluate models on the non-oversampled test set
models = [svm_grid, rf_grid, dt_grid, gb_model, dummy_model]
model_names = ['SVM', 'Random Forest', 'Decision Tree', 'Gradient Boosting', 'Dummy']
accuracies = []

for model, name in zip(models, model_names):
    print(f"\n{name} Model:")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{name} Confusion Matrix (n={len(y_test)})")
    plt.show()

# Compare model performances
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.show()

