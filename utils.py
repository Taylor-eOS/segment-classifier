import os

def convert_time_to_seconds(time_str):
    parts = [float(p) for p in time_str.split(':')]
    if len(parts) == 2:
        minutes, seconds = parts
        total_seconds = minutes * 60 + seconds
    elif len(parts) == 1:
        total_seconds = parts[0]
    else:
        raise ValueError("Invalid time format")
    return total_seconds

def convert_mp3(mp3_path, wav_path):
    if not os.path.exists(wav_path):
        print(f"Converting {mp3_path} to {wav_path}")
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")

def process_audio_files(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp3'):
            mp3_path = os.path.join(input_folder, filename)
            wav_filename = filename.replace('.mp3', '.wav')
            wav_path = os.path.join(input_folder, wav_filename)
            convert_mp3(mp3_path, wav_path)
