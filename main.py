import os
import shutil
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import csv

TRAINING = False
SR = 11025
HOP_LENGTH = 2205
HALF_WINDOW = 32
FRAMES_PER_SECOND = SR // HOP_LENGTH
WINDOW_SIZE = 2 * HALF_WINDOW + 1
EPOCHS = 8
BATCH_SIZE = 16
N_MFCC = 13
MODEL_DIR = Path('model')
INPUT_FOLDER = Path('input')
SEGMENTS_FILE = 'segments.txt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    mfcc_reshaped = mfcc.reshape(-1, 3, N_MFCC)
    mfcc_avg = mfcc_reshaped.mean(axis=1)
    return mfcc_avg

class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return self.features.shape[0]
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads=4, key_dim=32, ff_dim=128, num_transformer_blocks=2, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, num_heads * key_dim)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=num_heads * key_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='relu'
            )
            for _ in range(num_transformer_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_heads * key_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear(x)
        x = x.permute(1, 0, 2)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = x.permute(1, 0, 2)
        center_index = WINDOW_SIZE // 2
        x = x[:, center_index, :]
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x.squeeze()

def create_labels(timestamps, num_frames):
    times_in_seconds = np.array(timestamps)
    frame_indices = (times_in_seconds * FRAMES_PER_SECOND).astype(int)
    labels = np.zeros(num_frames, dtype=np.float32)
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

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_correct = 0
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_features.size(0)
            preds = (outputs >= 0.5).float()
            epoch_correct += (preds == batch_labels).sum().item()
        epoch_loss /= len(dataloader.dataset)
        epoch_acc = epoch_correct / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

def predictions_to_timestamps(predictions, threshold=0.5, frame_rate=FRAMES_PER_SECOND, window_size=WINDOW_SIZE):
    binary_predictions = (predictions >= threshold).astype(int)
    timestamps = []
    start_time = None
    for i, pred in enumerate(binary_predictions):
        current_time = i / frame_rate
        if pred == 1 and start_time is None:
            start_time = current_time - (HALF_WINDOW / frame_rate)
            if start_time < 0:
                start_time = 0
        elif pred == 0 and start_time is not None:
            end_time = current_time + (HALF_WINDOW / frame_rate)
            timestamps.append((start_time, end_time))
            start_time = None
    if start_time is not None:
        end_time = len(binary_predictions) / frame_rate
        timestamps.append((start_time, end_time))
    return timestamps

def infer(file_path, model_path, threshold=0.5):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = TransformerClassifier(input_dim=N_MFCC + 1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    features = extract_features(file_path)
    num_frames = features.shape[0]
    total_seconds = num_frames // FRAMES_PER_SECOND
    time_steps = np.arange(total_seconds)
    X_list = []
    position_indices = np.linspace(0, 1, WINDOW_SIZE).reshape(WINDOW_SIZE, 1)
    for sec in time_steps:
        i = sec * FRAMES_PER_SECOND
        start_idx = i - HALF_WINDOW
        end_idx = i + HALF_WINDOW + 1
        if start_idx < 0 or end_idx > num_frames:
            continue
        window_features = features[start_idx:end_idx]
        if window_features.shape[0] != WINDOW_SIZE:
            continue
        window_positions = position_indices
        window_features_with_pos = np.hstack((window_features, window_positions))
        X_list.append(window_features_with_pos)
    X = np.array(X_list)
    if X.shape[0] == 0:
        return None, None
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    timestamps = predictions_to_timestamps(predictions, threshold=threshold, frame_rate=FRAMES_PER_SECOND, window_size=WINDOW_SIZE)
    return predictions, timestamps

def main():
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Assuming utils is defined elsewhere
    import utils
    utils.process_audio_files(INPUT_FOLDER)
    segments_path = INPUT_FOLDER / SEGMENTS_FILE
    segments_dict = utils.parse_segments(segments_path)
    all_X = []
    all_y = []
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith('.wav'):
            continue
        file_path = INPUT_FOLDER / filename
        feature_file = MODEL_DIR / f'{filename}_features.npy'
        label_file = MODEL_DIR / f'{filename}_labels.npy'
        if feature_file.exists() and label_file.exists():
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
        position_indices = np.linspace(0, 1, WINDOW_SIZE).reshape(WINDOW_SIZE, 1)
        for sec in time_steps:
            i = sec * FRAMES_PER_SECOND
            start_idx = i - HALF_WINDOW
            end_idx = i + HALF_WINDOW + 1
            if start_idx < 0 or end_idx > num_frames:
                continue
            window_features = features[start_idx:end_idx]
            if window_features.shape[0] != WINDOW_SIZE:
                continue
            window_positions = position_indices
            window_features_with_pos = np.hstack((window_features, window_positions))
            center_label = labels[i]
            X_list.append(window_features_with_pos)
            y_list.append(center_label)
        X = np.array(X_list)
        y = np.array(y_list)
        if X.shape[0] == 0:
            continue
        all_X.append(X)
        all_y.append(y)
    if not all_X:
        return
    combined_X = np.concatenate(all_X, axis=0)
    combined_y = np.concatenate(all_y, axis=0)
    dataset = AudioDataset(combined_X, combined_y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model = TransformerClassifier(input_dim=N_MFCC + 1).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    train_model(model, dataloader, criterion, optimizer, DEVICE)
    model_save_path = MODEL_DIR / 'combined_model.pth'
    torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':
    if TRAINING:
        main()
    else:
        model_path = MODEL_DIR / 'combined_model.pth'
        audio_file = 'inference_input.wav'
        predictions, timestamps = infer(audio_file, model_path, threshold=0.5)
        if predictions is not None:
            np.savetxt('predictions.txt', predictions, fmt='%.10f')
            with open('timestamps.csv', 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Segment', 'Start (s)', 'End (s)'])
                for idx, (start, end) in enumerate(timestamps):
                    csv_writer.writerow([idx+1, f"{start:.2f}", f"{end:.2f}"])
        else:
            print("No segments detected.")

