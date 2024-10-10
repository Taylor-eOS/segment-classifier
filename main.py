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

def parse_segments(segments_file_path):
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

def extract_features(file_path, n_mfcc, hop_length, sr):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc = mfcc.T
    return mfcc

def positional_encoding(max_length, d_model):
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

def build_model(input_shape, num_heads=4, key_dim=32, ff_dim=128, num_transformer_blocks=2, dropout_rate=0.1):
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
    center_index = input_shape[0] // 2
    center_output = Lambda(lambda x: x[:, center_index, :])(x)
    outputs = Dense(1, activation='sigmoid', name='output')(center_output)
    model = Model(inputs=features_input, outputs=outputs)
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def add_positional_encoding(feature_matrix):
    max_length, n_mfcc = feature_matrix.shape
    pos_encoding = positional_encoding(max_length, d_model=n_mfcc)
    pos_encoded_features = feature_matrix + pos_encoding
    return pos_encoded_features

def create_labels(timestamps, num_frames, hop_length, sr):
    times_in_seconds = np.array(timestamps)
    frames_per_second = sr / hop_length
    frame_indices = (times_in_seconds * frames_per_second).astype(int)
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
    epochs = 10
    model.fit(dataset, epochs=epochs)

def main():
    input_folder = 'input'
    model_dir = 'model'
    segments_file = 'segments.txt'
    sr = 22050
    n_mfcc = 13
    hop_length = 525
    window_duration = 60
    frames_per_second = sr // hop_length
    window_size = window_duration * frames_per_second
    half_window = window_size // 2
    os.makedirs(model_dir, exist_ok=True)
    utils.process_audio_files(input_folder)
    segments_dict = parse_segments(os.path.join(input_folder, segments_file))
    shutil.rmtree(model_dir) #Comment out when keeping the same structure

    for filename in os.listdir(input_folder):
        if not filename.endswith('.wav'):
            continue
        file_path = os.path.join(input_folder, filename)
        feature_file = f'{model_dir}/{filename}_features.npy'
        label_file = f'{model_dir}/{filename}_labels.npy'
        if os.path.exists(feature_file) and os.path.exists(label_file):
            features = np.load(feature_file)
            labels = np.load(label_file)
        else:
            features = extract_features(file_path, n_mfcc, hop_length, sr)
            timestamps = segments_dict.get(filename, [])
            num_frames = features.shape[0]
            labels = create_labels(timestamps, num_frames, hop_length, sr)
            np.save(feature_file, features)
            np.save(label_file, labels)
        num_frames = features.shape[0]
        frames_per_second = sr // hop_length
        total_seconds = num_frames // frames_per_second
        time_steps = np.arange(total_seconds)
        X_list = []
        y_list = []
        for sec in time_steps:
            i = sec * frames_per_second
            i = i % num_frames
            start_idx = i - half_window
            end_idx = i + half_window
            indices = np.arange(start_idx, end_idx) % num_frames
            window_features = features[indices]
            if window_features.shape[0] != window_size:
                print(f"Warning: Window size mismatch for {filename} at second {sec}. Expected {window_size}, got {window_features.shape[0]}")
                continue
            window_features = add_positional_encoding(window_features)
            center_label = labels[i]
            X_list.append(window_features)
            y_list.append(center_label)
        X = np.array(X_list)
        y = np.array(y_list)
        print(f"Filename: {filename}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        input_shape = (window_size, n_mfcc)
        model = build_model(input_shape)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(16).prefetch(tf.data.AUTOTUNE)
        train_model(model, dataset)
        model.save(f'{model_dir}/{filename}_model.h5')

if __name__ == '__main__':
    main()

