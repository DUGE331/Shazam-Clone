import librosa.display
import numpy as np
import sounddevice as sd
import librosa
import scipy.ndimage
import hashlib
import matplotlib.pyplot as plt
import pickle 
import os
import soundcard as sc
import soundfile as sf


#load audio
y, sr = librosa.load(r"c:\Users\strep\Downloads\Hanumankind Kalmi Hanumankind – Big Dawgs Prod.Kalmi Official Music Video Def Jam India.mp3", sr=None)
song_path = r"c:\Users\strep\Downloads\Hanumankind Kalmi Hanumankind – Big Dawgs Prod.Kalmi Official Music Video Def Jam India.mp3"
song_name = os.path.basename(song_path)  
song_id = os.path.splitext(song_name)[0]
fingerprint_file = f"fingerprints/{song_id}.pk1"

#spectrogram
S = np.abs(librosa.stft(y))
S_db = librosa.amplitude_to_db(S, ref=np.max)

plt.figure(figsize=(10, 6))
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         sr=sr, y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectogram (dB)')
#plt.show()
print("S_db max:", np.max(S_db))
print("S_db mean:", np.mean(S_db))
print("S_db min:", np.min(S_db))

#Detecting Highpoints (Peaks) in the Spectrogram shows peak cooridantes
def find_peaks(S_db, threshold=-25): #20 changed threshold
    neighborhood_size = (60, 60) #size of region searching for peaks
    print(f"Using neighborhood_size: {neighborhood_size}")
    local_max = scipy.ndimage.maximum_filter(S_db, size=neighborhood_size)
    detected_peaks = local_max.astype(bool) & (S_db > threshold).astype(bool)  # Detect peaks above the threshold
    peaks = np.argwhere(detected_peaks)
    print(f"Detected {len(peaks)} peaks with threshold {threshold}")
    return peaks

peaks = find_peaks(S_db)

print(f"Found {len(peaks)} peaks")

#learn this 
#Creating Fingerprints by Hashing Peak Pairs (collectin two peaks in one dataset)
def generate_fingerprints(peaks,sr,hop_length=512, fan_out=5):
    fingerprints =[]

    for i in range(len(peaks)):
        freq1, time1 = peaks[i] #location of the peak

        for j in range(1, fan_out): #finds nearest peaks to J=0
            if i + j < len(peaks):
                freq2, time2 = peaks[i + j]
                delta_t = time2 - time1
                if 0 < delta_t <200: #diffrence in time between peaks within 200hops
                    hash_input = f"{freq1}|{freq2}|{delta_t}" #freq and time diffrence of peaks
                    hash_val = hashlib.sha1(hash_input.encode()).hexdigest()

                    time1_sec = librosa.frames_to_time(time1, sr=sr, hop_length=hop_length)
                    fingerprints.append((hash_val, time1_sec))

    return fingerprints

fingerprints = generate_fingerprints(peaks, sr)

#Save Fingerprints to a File
os.makedirs("fingerprints", exist_ok=True)
fingerprint_file = f"fingerprints/{song_id}.pk1"

with open(fingerprint_file, "wb") as f:
    pickle.dump(fingerprints, f)
    print(f"Saved {len(fingerprints)} fingerprints to {fingerprint_file}")

with open(fingerprint_file, "wb") as f:
    pickle.dump(fingerprints, f) #saves binary form in file

    print(f"Saved {len(fingerprints)} fingerprints to {fingerprint_file}")

#Fingerprint file storage and naming
def build_database(fingerprint_dir="fingerprints"):
    database = {}

    for file in os.listdir(fingerprint_dir):
        if file.endswith(".pk1"):
            song_id = file.replace(".pk1", "")
            with open(os.path.join(fingerprint_dir, file), "rb") as f:
                fingerprints = pickle.load(f)

                for hash_val, time in fingerprints:
                    if hash_val not in database:
                        database[hash_val] = []
                    database[hash_val].append((song_id, time))

    return database

database = build_database()
with open("fingerprint_database.pk1", "wb") as f:
    pickle.dump(database, f)

OUTPUT_FILE_NAME = "recorded_audio.wav"
SAMPLE_RATE = 44100
RECORD_SECONDS = 5

#record internal audio
def record_audio_loopback(output_file=OUTPUT_FILE_NAME, duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    print("Recording system audio using loopback...")

    default_speaker = sc.default_speaker()

    with sc.get_microphone(id=str(default_speaker.name), include_loopback=True).recorder(samplerate=sample_rate) as mic:
        data = mic.record(numframes=sample_rate * duration)
        print("Recording done.")

#convert to mono

        if data.shape[1] > 1:
            data = data.mean(axis=1)

#save to .wav

        sf.write(file=output_file, data=data, samplerate=sample_rate)
        print(f"Saved recording to {output_file}")
        return output_file

#load audio and spectogram

def process_recorded_audio(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    print(f"Loaded audio: {y.shape}, Sample rate: {sr}")

    S = np.abs(librosa.stft(y))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    print(f"Spectrogram shape: {S_db.shape}")
    return S_db, sr

fingerprints_db = {}

#fingerprint finding record
def fingerprint_recording(spectogram_db, sr):
    peaks= find_peaks(spectogram_db, threshold=-25)
    print(f"Found {len(peaks)} peaks in recording")

    fingerprints = generate_fingerprints(peaks, sr)
    print(f"Generated {len(fingerprints)} fingerprints from recording")
    return fingerprints


#testing finerprint matching
fingerprint_folder = "fingerprints"
for filename in os.listdir(fingerprint_folder):
    if filename.endswith(".pk1"):
        song_id = os.path.splitext(filename)[0]
        with open(os.path.join(fingerprint_folder, filename), "rb") as f:
            fingerprints_db[song_id] = pickle.load(f)
print(f"Loaded fingerprints for {len(fingerprints_db)} songs")


#fingerprint matching

def match_fingerprints(recorded_fingerprints, database_path="fingerprint_database.pk1"):
    with open(database_path, "rb") as f:
        database = pickle.load(f)

    match_counts = {}
    for hash_val, recorded_time in recorded_fingerprints:
        if hash_val in database:
            for song_id, song_time in database[hash_val]:
                if song_id not in match_counts:
                    match_counts[song_id] = 0
                match_counts[song_id] +=1

    if not match_counts:
        print("No match found.")
        return None
    
    best_match= max(match_counts.items(), key=lambda item: item[1])
    print(f"Best Match: {best_match[0]} with {best_match[1]} matching fingerprints")
    return best_match

if __name__ == "__main__":
    wav_file = record_audio_loopback()
    S_db, sr = process_recorded_audio(wav_file)
    fingerprints = fingerprint_recording(S_db, sr)
    match_fingerprints(fingerprints)

#use mp3 files in folder later
