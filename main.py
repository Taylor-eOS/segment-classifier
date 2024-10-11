import os
import re
import shutil
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import utils

# =========================
# Global Configuration
# =========================
SR = 22050               # Sampling rate
N_MFCC = 13              # Number of MFCC features
HOP_LENGTH = 4410        # Hop length to achieve 5 frames per second
FRAMES_PER_SECOND = SR // HOP_LENGTH  # Frames per second (5)
WINDOW_DURATION = 9      # Window duration in seconds (center second + 4 on each side)
WINDOW_SIZE = WINDOW_DURATION * FRAMES_PER_SECOND  # Total window size in frames (45)
HALF_WINDOW = WINDOW_SIZE // 2  # Half window size (22)
MODEL_DIR = 'model'      # Directory to save models
INPUT_FOLDER = 'input'   # Directory containing input files
SEGMENTS_FILE = 'segments.txt'  # Segments file name

# =========================
# Function Definitions
# =========================

def parse_segments(segments_file_path):
    """
    Parses the segments file and returns a dictionary mapping filenames to timestamp lists.
    """
    segments_dict = {}
    with open(segments_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 6):
            header = lines[i].strip()
            filename_match = re.match(r'\[(.*?)\]', header)
            if filename_match:
                filename = filename_match.group(1)
                timestamps = []
                for j in range(1, 6):
                    time_range = lines[i+j].strip()
                    start, end = time_range.split('-')
                    start_sec = utils.convert_time_to_seconds(start)
                    end_sec = utils.convert_time_to_seconds(end)
                    timestamps.extend([start_sec, end_sec])
                segments_dict[filename] = timestamps
    return segments_dict

def extract_features(file_path):
    """
    Extracts MFCC features from an audio file.
    """
    y, sr = librosa.load(file_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    mfcc = mfcc.T  # Transpose to shape (frames, n_mfcc)
    return mfcc

def positional_encoding(max_length, d_model):
    """
    Generates positional encoding for transformer models.
    """
    pos = np.arange(max_length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.zeros((max_length, d_model))
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = cosines
    return pos_encoding

def build_model():
    """
    Builds and compiles the Transformer-based model.
    """
    num_heads = 4
    key_dim = 32
    ff_dim = 128
    num_transformer_blocks = 2
    dropout_rate = 0.1
    input_shape = (WINDOW_SIZE, N_MFCC)

    def transformer_block(x):
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate
        )(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
        ffn_output = Dense(ff_dim, activation='relu')(out1)
        ffn_output = Dense(key_dim * num_heads)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        return out2

    features_input = Input(shape=input_shape, name='features_input')
    x = LayerNormalization(epsilon=1e-6)(features_input)
    x = Dense(num_heads * key_dim)(x)
    for _ in range(num_transformer_blocks):
        x = transformer_block(x)
    center_index = WINDOW_SIZE // 2
    x = Lambda(lambda x: x[:, center_index, :])(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs=features_input, outputs=outputs)
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def add_positional_encoding(feature_matrix):
    """
    Adds positional encoding to the feature matrix.
    """
    pos_encoding = positional_encoding(WINDOW_SIZE, N_MFCC)
    pos_encoded_features = feature_matrix + pos_encoding
    return pos_encoded_features

def create_labels(timestamps, num_frames):
    """
    Creates binary labels based on segment timestamps.
    """
    times_in_seconds = np.array(timestamps)
    frame_indices = (times_in_seconds * FRAMES_PER_SECOND).astype(int)
    labels = np.zeros(num_frames, dtype=np.int32)
    current_label = 0
    prev_frame = 0
    for frame in frame_indices:
        if frame >= num_frames:
            frame = frame % num_frames
        labels[prev_frame:frame] = current_label
        current_label = 1 - current_label
        prev_frame = frame
    if prev_frame < num_frames:
        labels[prev_frame:] = current_label
    return labels

def train_model(model, dataset):
    """
    Trains the model on the provided dataset.
    """
    epochs = 10
    model.fit(dataset, epochs=epochs)

# =========================
# Main Execution
# =========================

def main():
    # Ensure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Process audio files using utils
    utils.process_audio_files(INPUT_FOLDER)

    # Parse the segments file
    segments_path = os.path.join(INPUT_FOLDER, SEGMENTS_FILE)
    segments_dict = parse_segments(segments_path)

    # Optionally remove the model directory (uncomment if needed)
    # shutil.rmtree(MODEL_DIR)  # Comment out when keeping the same structure

    # Iterate over each audio file in the input folder
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith('.wav'):
            continue  # Skip non-WAV files

        file_path = os.path.join(INPUT_FOLDER, filename)
        feature_file = os.path.join(MODEL_DIR, f'{filename}_features.npy')
        label_file = os.path.join(MODEL_DIR, f'{filename}_labels.npy')

        # Load existing features and labels if available
        if os.path.exists(feature_file) and os.path.exists(label_file):
            features = np.load(feature_file)
            labels = np.load(label_file)
        else:
            # Extract features and create labels
            features = extract_features(file_path)
            timestamps = segments_dict.get(filename, [])
            num_frames = features.shape[0]
            labels = create_labels(timestamps, num_frames)
            # Save features and labels for future use
            np.save(feature_file, features)
            np.save(label_file, labels)

        num_frames = features.shape[0]
        total_seconds = num_frames // FRAMES_PER_SECOND
        time_steps = np.arange(total_seconds)

        X_list = []
        y_list = []

        # Generate windows centered around each second
        for sec in time_steps:
            i = sec * FRAMES_PER_SECOND
            i = i % num_frames  # Wrap around if necessary
            start_idx = i - HALF_WINDOW
            end_idx = i + HALF_WINDOW
            indices = np.arange(start_idx, end_idx) % num_frames
            window_features = features[indices]

            if window_features.shape[0] != WINDOW_SIZE:
                print(f"Warning: Window size mismatch for {filename} at second {sec}. Expected {WINDOW_SIZE}, got {window_features.shape[0]}")
                continue  # Skip incomplete windows

            # Add positional encoding
            window_features = add_positional_encoding(window_features)

            # Get the label for the center frame
            center_label = labels[i]
            X_list.append(window_features)
            y_list.append(center_label)

        # Convert lists to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)

        print(f"Filename: {filename}")
        print(f"X shape: {X.shape}")  # Expected: (num_windows, 45, 13)
        print(f"y shape: {y.shape}")  # Expected: (num_windows,)

        # Build and compile the model
        model = build_model()

        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(16).prefetch(tf.data.AUTOTUNE)

        # Train the model
        train_model(model, dataset)

        # Save the trained model
        model_save_path = os.path.join(MODEL_DIR, f'{filename}_model.h5')
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}\n")

if __name__ == '__main__':
    main()

