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

SR = 11025
HOP_LENGTH = 2205 #Make this values where a third also divides through the SR without rest
HALF_WINDOW = 32
FRAMES_PER_SECOND = SR // HOP_LENGTH
WINDOW_SIZE = 2 * HALF_WINDOW + 1
EPOCHS = 12
BATCH_SIZE = 16
N_MFCC = 13
MODEL_DIR = 'model'
INPUT_FOLDER = 'input'
SEGMENTS_FILE = 'segments.txt'

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    new_hop_length = HOP_LENGTH // 3
    if HOP_LENGTH % 3 != 0:
        raise ValueError("HOP_LENGTH must be divisible by 3 to evenly average frames.")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=new_hop_length)
    mfcc = mfcc.T
    num_frames = mfcc.shape[0]
    remainder = num_frames % 3
    if remainder != 0:
        mfcc = mfcc[:num_frames - remainder, :]
    mfcc_reshaped = mfcc.reshape(-1, 3, N_MFCC)  # Shape: (num_new_frames, 3, N_MFCC)
    mfcc_avg = mfcc_reshaped.mean(axis=1)        # Shape: (num_new_frames, N_MFCC)
    return mfcc_avg

def build_model():
    num_heads = 4
    key_dim = 32
    ff_dim = 128
    num_transformer_blocks = 2
    dropout_rate = 0.1
    input_shape = (WINDOW_SIZE, N_MFCC + 1)

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

def create_labels(timestamps, num_frames):
    times_in_seconds = np.array(timestamps)
    frame_indices = (times_in_seconds * FRAMES_PER_SECOND).astype(int)
    labels = np.zeros(num_frames, dtype=np.int32)
    current_label = 0
    prev_frame = 0
    for frame in frame_indices:
        if frame >= num_frames:
            frame = num_frames
        labels[prev_frame:frame] = current_label
        current_label = 1 - current_label
        prev_frame = frame
    if prev_frame < num_frames:
        labels[prev_frame:] = current_label
    return labels

def train_model(model, dataset):
    model.fit(dataset, epochs=EPOCHS)

def main():
    # Remove and recreate the MODEL_DIR
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    os.makedirs(MODEL_DIR)
    # Process audio files
    utils.process_audio_files(INPUT_FOLDER)
    segments_path = os.path.join(INPUT_FOLDER, SEGMENTS_FILE)
    segments_dict = utils.parse_segments(segments_path)
    all_X = []
    all_y = []
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith('.wav'):
            continue 
        file_path = os.path.join(INPUT_FOLDER, filename)
        feature_file = os.path.join(MODEL_DIR, f'{filename}_features.npy')
        label_file = os.path.join(MODEL_DIR, f'{filename}_labels.npy')
        if os.path.exists(feature_file) and os.path.exists(label_file):
            features = np.load(feature_file)
            labels = np.load(label_file)
        else:
            features = extract_features(file_path)
            timestamps = segments_dict.get(filename, [])
            num_frames = features.shape[0]
            labels = create_labels(timestamps, num_frames)
            np.save(feature_file, features)
            np.save(label_file, labels)
        num_frames = features.shape[0]
        total_seconds = num_frames // FRAMES_PER_SECOND
        time_steps = np.arange(total_seconds)
        X_list = []
        y_list = []
        # Precompute position indices
        position_indices = np.linspace(0, 1, WINDOW_SIZE).reshape(WINDOW_SIZE, 1)  # Normalized positions
        for sec in time_steps:
            i = sec * FRAMES_PER_SECOND
            start_idx = i - HALF_WINDOW
            end_idx = i + HALF_WINDOW + 1
            if start_idx < 0 or end_idx > num_frames:
                continue
            window_features = features[start_idx:end_idx]
            if window_features.shape[0] != WINDOW_SIZE:
                print(f"Warning: Window size mismatch for {filename} at second {sec}. Expected {WINDOW_SIZE}, got {window_features.shape[0]}")
                continue
            # Add position dimension
            window_positions = position_indices  # Shape: (WINDOW_SIZE, 1)
            window_features_with_pos = np.hstack((window_features, window_positions))
            center_label = labels[i]
            X_list.append(window_features_with_pos)
            y_list.append(center_label)
        X = np.array(X_list)
        y = np.array(y_list)
        print(f"Filename: {filename}")
        print(f"Total Seconds: {total_seconds}")
        print(f"Valid Windows: {X.shape[0]}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        if X.shape[0] == 0:
            print(f"No valid windows for {filename}. Skipping.\n")
            continue
        # Aggregate data
        all_X.append(X)
        all_y.append(y)
    # Combine all data
    if not all_X:
        print("No data available for training.")
        return
    combined_X = np.concatenate(all_X, axis=0)
    combined_y = np.concatenate(all_y, axis=0)
    print(f"Combined X shape: {combined_X.shape}")
    print(f"Combined y shape: {combined_y.shape}")
    # Build and train a single model
    model = build_model()
    dataset = tf.data.Dataset.from_tensor_slices((combined_X, combined_y))
    dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_model(model, dataset)
    # Save the single trained model
    model_save_path = os.path.join(MODEL_DIR, 'combined_model.h5')
    model.save(model_save_path)
    print(f"Combined model saved to {model_save_path}\n")

if __name__ == '__main__':
    main()

