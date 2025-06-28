import librosa
import numpy as np
import os

sr = 22050

def get_audio_path(track_id, base_dir=os.path.join("data", "genres_original")):
    '''
    Returns the file path to the 30s audio track based on the provided track_id.

    Parameters:
        track_id (int or str): The unique ID of the track.
        base_dir (str): The base directory where the audio files are stored.

    Returns:
        str: Path to the audio file.
    '''
    genre_dir = track_id.split('.')[0]
    return os.path.join(base_dir, genre_dir, track_id)


def fix_length(y, sr):
    '''
    Ensures that all audio tracks are 30s long.
    If the track is longer, it will be clipped.
    If the track is shorter, it will be zero-padded.

    Parameters:
        y (ndarray): The audio time series.
        sr (int): The sampling rate of the audio.

    Returns:
        ndarray: Audio time series of exactly 30s.
    '''
    target_length = 30 * sr
    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        pad_length = target_length - len(y)
        y = np.pad(y, (0, pad_length), mode='constant')
    return y


def divide_into_segments(y, save=False, track_dir=None, track_id=None, genre=None, valid_tracks=None, valid_genres=None, valid_paths=None):
    '''
    Segments a 30s audio track into 3s segments.
    Computes log-scaled mel spectrograms for each segment and normalizes them per-sample.

    If save is True, saves each segment's spectrogram as a .npy file in the specified directory.
    If save is False, returns the spectrograms as a python list.

    Parameters:
        y (ndarray): The audio time series of a track.
        save (bool): Whether to save the spectrograms or return them.
        track_dir (str): Directory where spectrogram files should be saved (required if save=True).
        track_id (int or str): The ID of the track (required if save=True).
        genre (int): Encoded genre label (required if save=True).
        valid_tracks (list): List to store track IDs (required if save=True).
        valid_genres (list): List to store genre labels (required if save=True).
        valid_paths (list): List to store file paths of saved spectrograms (required if save=True).

    Returns:
        list: List of spectrograms (only if save=False).
    '''
    segments = []
    clip_duration = 3 
    samples_per_clip = clip_duration * sr
    total_segments = len(y) // samples_per_clip

    for i in range(total_segments):
        start_sample = i * samples_per_clip
        end_sample = start_sample + samples_per_clip
        y_segment = y[start_sample:end_sample]

        m = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        m_dB = librosa.power_to_db(m, ref=np.max)
 
        #m_dB = (m_dB - np.mean(m_dB)) / (np.std(m_dB) + 0.0001)
        
        if save:
            segment_filename = f"{track_dir}/track_{track_id}_segment_{i}.npy"
            np.save(segment_filename, m_dB)

            valid_tracks.append(track_id)
            valid_genres.append(genre)
            valid_paths.append(segment_filename)
        else:
            segments.append(m_dB)
            
    if not save:
        return segments