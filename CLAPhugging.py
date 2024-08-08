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

results = []

for entry in data:
    original_work = entry['Original Work']
    second_song = entry['Second Song']

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
    
    results.append({
        "Original Work": original_work,
        "Second Song": second_song,
        "Similarity": similarity
    })

# Save embeddings to a file for future use
with open(embeddings_file, 'w') as f:
    json.dump(embeddings_dict, f)

for result in results:
    print(f"Original Work: {result['Original Work']}, Second Song: {result['Second Song']}, Similarity: {result['Similarity']}")

df_results = pd.DataFrame(results)
print(df_results)
