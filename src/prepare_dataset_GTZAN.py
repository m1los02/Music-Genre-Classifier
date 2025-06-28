import pandas as pd
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from data_preparation_GTZAN import get_audio_path, fix_length, divide_into_segments


def save_spectrograms(track_ids, genre_encoded, output_base_dir, csv_name):
    '''
    Processes a list of audio tracks by:
        - Loading each audio file.
        - Fixing their lengths to 30s.
        - Segmenting them into 3s segments.
        - Saving mel spectrograms of each segment.
        - Creating a CSV file containing metadata: track ID, genre label, and segment file paths.

    Parameters:
        track_ids (list): List of track IDs to process.
        genre_encoded (list): Genre labels for each track.
        output_base_dir (str): Base directory where the spectrograms will be saved.
        csv_name (str): Name of the output CSV file.
    '''
    valid_tracks = []
    valid_paths = []
    valid_genres = []
    for track_id, genre in zip(track_ids, genre_encoded):
        try:
            y, sr = librosa.load(get_audio_path(track_id), sr=22050) 
            y = fix_length(y, sr)
            track_dir = os.path.join(output_base_dir, f"track_{track_id}")
            os.makedirs(track_dir, exist_ok=True)
            divide_into_segments(y, True, track_dir, track_id, genre, valid_tracks, valid_genres, valid_paths)
        except Exception as e:
            print(f"Error loading track {track_id}: {e}")
            
    formatted_track_ids = [f"track_{track_id}" for track_id in valid_tracks]
    final_df = pd.DataFrame({
        'track_id': formatted_track_ids,
        'genre_id': valid_genres,
        'segment_paths': valid_paths,
    })
    final_df.to_csv(csv_name, index=False)


tracks = pd.read_csv('data/features_30_sec.csv')
track_ids = tracks['filename'].to_numpy()
genre_labels = tracks['label'].to_numpy()

label_encoder = LabelEncoder()
genre_encoded = label_encoder.fit_transform(genre_labels)

track_ids_train, track_ids_rest, genre_encoded_train, genre_encoded_rest = train_test_split(
    track_ids, genre_encoded, test_size=0.2, stratify=genre_encoded, random_state=42)

track_ids_val, track_ids_test, genre_encoded_val, genre_encoded_test = train_test_split(
    track_ids_rest, genre_encoded_rest, test_size=0.5, stratify=genre_encoded_rest, random_state=42)

genre_id_to_name = {i: name for i, name in enumerate(label_encoder.classes_)}
with open("data/genre_id_to_name.json", "w") as f:
    json.dump(genre_id_to_name, f, indent=4) 

save_spectrograms(track_ids_train, genre_encoded_train, 'data/mel_spectrograms_train' , 'data/train.csv')
save_spectrograms(track_ids_val, genre_encoded_val, 'data/mel_spectrograms_val' , 'data/val.csv')
save_spectrograms(track_ids_test, genre_encoded_test, 'data/mel_spectrograms_test' , 'data/test.csv')