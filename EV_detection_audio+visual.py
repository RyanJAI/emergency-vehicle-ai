# ============================================================
# COMPLETE EMERGENCY VEHICLE DETECTION SYSTEM - FIXED
# NOW INCLUDES FLAC FILES IN TRAINING!
# ============================================================




!pip install ultralytics librosa tensorflow opencv-python soundfile matplotlib -q




import os
import cv2
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from collections import deque
from google.colab import drive, files
from IPython.display import display, Video, Audio
import subprocess
import zipfile
import shutil
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')




print("="*70)
print("🚨 EMERGENCY VEHICLE DETECTION SYSTEM - FIXED")
print("NOW INCLUDES FLAC FILES IN TRAINING!")
print("="*70)




# ============================================================
# STEP 1: MOUNT DRIVE AND SETUP
# ============================================================
drive.mount('/content/drive', force_remount=True)




# Paths
POSITIVE_AUDIO_PATH = "/content/audio_data/positive_ev_audios"
NEGATIVE_AUDIO_PATH = "/content/audio_data/negative_ev_audio"
VISION_MODEL_PATH = "/content/drive/MyDrive/police_light_model.pt"




os.makedirs(POSITIVE_AUDIO_PATH, exist_ok=True)
os.makedirs(NEGATIVE_AUDIO_PATH, exist_ok=True)




print("\n📂 Training folders ready")




# ============================================================
# STEP 2: FFMPEG PROCESSING FUNCTION
# ============================================================
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
TARGET_LENGTH = SAMPLE_RATE * DURATION




def process_with_ffmpeg(input_path, output_path):
    cmd = f"ffmpeg -i '{input_path}' -ac 1 -ar {SAMPLE_RATE} '{output_path}' -y -loglevel error"
    subprocess.run(cmd, shell=True)
    return output_path if os.path.exists(output_path) else None




def extract_features(file_path):
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(signal) < TARGET_LENGTH:
            signal = np.pad(signal, (0, TARGET_LENGTH - len(signal)))
        else:
            signal = signal[:TARGET_LENGTH]
       
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        return None




# ============================================================
# STEP 3: COUNT FILES - FIXED TO INCLUDE FLAC!
# ============================================================
def count_files(folder):
    if os.path.exists(folder):
        return len([f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.flac'))])
    return 0




def count_all_files(folder):
    if os.path.exists(folder):
        all_files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.flac'))]
        wav = len([f for f in all_files if f.endswith('.wav')])
        mp3 = len([f for f in all_files if f.endswith('.mp3')])
        flac = len([f for f in all_files if f.endswith('.flac')])
        return len(all_files), wav, mp3, flac
    return 0, 0, 0, 0




# Show current counts
total_pos, pos_wav, pos_mp3, pos_flac = count_all_files(POSITIVE_AUDIO_PATH)
total_neg, neg_wav, neg_mp3, neg_flac = count_all_files(NEGATIVE_AUDIO_PATH)




print(f"\n📊 Current training data:")
print(f"   Positive (sirens): {total_pos} total (WAV: {pos_wav}, MP3: {pos_mp3}, FLAC: {pos_flac})")
print(f"   Negative (non-sirens): {total_neg} total (WAV: {neg_wav}, MP3: {neg_mp3}, FLAC: {neg_flac})")




# ============================================================
# STEP 4: UPLOAD TRAINING DATA IF NEEDED
# ============================================================
def extract_zip_and_process(zip_path, target_folder):
    temp_dir = "/content/temp_extract"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
   
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
   
    count = 0
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.mp4', '.flac')):
                src = os.path.join(root, file)
                temp_processed = f"/content/temp_processed_{count}.wav"
                process_with_ffmpeg(src, temp_processed)
               
                if os.path.exists(temp_processed) and os.path.getsize(temp_processed) > 1000:
                    dst = os.path.join(target_folder, f"processed_{count}_{file.replace('.', '_')}.wav")
                    shutil.move(temp_processed, dst)
                    count += 1
   
    shutil.rmtree(temp_dir)
    return count




# Upload positives if needed
if total_pos == 0:
    print("\n📁 Upload POSITIVE ZIP (siren sounds):")
    pos_zip = files.upload()
    for zip_name in pos_zip.keys():
        count = extract_zip_and_process(zip_name, POSITIVE_AUDIO_PATH)
        print(f"✅ Processed {count} positive files")
        os.remove(zip_name)
    total_pos, _, _, _ = count_all_files(POSITIVE_AUDIO_PATH)




# Upload negatives if needed
if total_neg == 0:
    print("\n📁 Upload NEGATIVE ZIP (non-siren sounds):")
    neg_zip = files.upload()
    for zip_name in neg_zip.keys():
        count = extract_zip_and_process(zip_name, NEGATIVE_AUDIO_PATH)
        print(f"✅ Processed {count} negative files")
        os.remove(zip_name)
    total_neg, _, _, _ = count_all_files(NEGATIVE_AUDIO_PATH)




# Final counts
total_pos, pos_wav, pos_mp3, pos_flac = count_all_files(POSITIVE_AUDIO_PATH)
total_neg, neg_wav, neg_mp3, neg_flac = count_all_files(NEGATIVE_AUDIO_PATH)




print(f"\n📊 FINAL TRAINING DATA:")
print(f"   🔴 Positives (sirens):     {total_pos} (WAV:{pos_wav}, MP3:{pos_mp3}, FLAC:{pos_flac})")
print(f"   🟢 Negatives (non-sirens): {total_neg} (WAV:{neg_wav}, MP3:{neg_mp3}, FLAC:{neg_flac})")
print(f"   📊 Total samples:          {total_pos + total_neg}")




if total_pos == 0 or total_neg == 0:
    raise SystemExit("❌ Missing training data!")




# ============================================================
# STEP 5: TRAIN AUDIO MODEL
# ============================================================
print("\n" + "="*70)
print("🔊 TRAINING AUDIO MODEL")
print("="*70)




features, labels = [], []




print("Loading positive samples (sirens)...")
for f in os.listdir(POSITIVE_AUDIO_PATH):
    if f.endswith(('.wav', '.mp3', '.flac')):
        file_path = os.path.join(POSITIVE_AUDIO_PATH, f)
        feat = extract_features(file_path)
        if feat is not None:
            features.append(feat)
            labels.append(1)




print("Loading negative samples (non-sirens)...")
for f in os.listdir(NEGATIVE_AUDIO_PATH):
    if f.endswith(('.wav', '.mp3', '.flac')):
        file_path = os.path.join(NEGATIVE_AUDIO_PATH, f)
        feat = extract_features(file_path)
        if feat is not None:
            features.append(feat)
            labels.append(0)




X, y = np.array(features), np.array(labels)
print(f"Total: {len(X)} (Sirens: {sum(y)}, Non-sirens: {len(y)-sum(y)})")




X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)




mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / (std + 1e-6)
X_val = (X_val - mean) / (std + 1e-6)
X_test = (X_test - mean) / (std + 1e-6)




np.save("/content/train_mean.npy", mean)
np.save("/content/train_std.npy", std)
print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")




audio_model = Sequential([
    Dense(256, activation='relu', input_shape=(N_MFCC,), kernel_regularizer=l2(0.001)),
    BatchNormalization(), Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(), Dropout(0.4),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(32, activation='relu'), Dropout(0.3),
    Dense(1, activation='sigmoid')
])




audio_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
audio_model.summary()




print("\n🏋️ Training for 50 epochs...")
history = audio_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=50, batch_size=32,
                         callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                         verbose=1)




test_loss, test_acc = audio_model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Audio Model Test Accuracy: {test_acc:.2%}")




audio_model.save("/content/audio_model_final.h5")
!cp /content/audio_model_final.h5 /content/drive/MyDrive/
print("✅ Audio model saved to Drive")




plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Audio Model Accuracy')
plt.grid(True, alpha=0.3)




plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Audio Model Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




# ============================================================
# STEP 6: LOAD VISION MODEL
# ============================================================
print("\n" + "="*70)
print("🎥 LOADING VISION MODEL")
print("="*70)




vision_model = None
if os.path.exists(VISION_MODEL_PATH):
    vision_model = YOLO(VISION_MODEL_PATH)
    print("✅ Vision model loaded")
else:
    print("⚠️ Vision model not found. Audio-only mode.")




# ============================================================
# STEP 7: TEMPORAL PATTERN DETECTOR
# ============================================================




class TemporalPoliceDetector:
    def __init__(self, model, history_frames=30, flash_threshold=3):
        self.model = model
        self.history_frames = history_frames
        self.flash_threshold = flash_threshold
        self.detection_history = deque(maxlen=history_frames)
        self.confidence_history = deque(maxlen=history_frames)
        self.color_history = deque(maxlen=history_frames)
       
    def detect_colors(self, frame, box):
        if self.model is None:
            return False, False, False
        x1, y1, x2, y2 = map(int, box)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False, False, False
       
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
       
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
       
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
       
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
       
        total_pixels = roi.shape[0] * roi.shape[1]
        has_red = cv2.countNonZero(mask_red) > total_pixels * 0.05
        has_blue = cv2.countNonZero(mask_blue) > total_pixels * 0.05
        has_yellow = cv2.countNonZero(mask_yellow) > total_pixels * 0.1
       
        return has_red, has_blue, has_yellow
   
    def analyze_frame(self, frame, frame_idx, fps):
        if self.model is None:
            return {
                'has_detection': False,
                'current_confidence': 0,
                'has_red': False, 'has_blue': False, 'has_yellow': False,
                'flash_count': 0, 'is_flashing': False, 'is_police': False,
                'box_coords': None
            }
       
        results = self.model(frame, conf=0.4, verbose=False)
       
        has_detection = len(results[0].boxes) > 0
        max_conf = 0
        has_red = False
        has_blue = False
        has_yellow = False
        box_coords = None
       
        if has_detection:
            best_box = None
            max_conf = 0
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf > max_conf:
                    max_conf = conf
                    best_box = box.xyxy[0]
           
            if best_box is not None:
                box_coords = best_box
                has_red, has_blue, has_yellow = self.detect_colors(frame, best_box)
       
        is_valid = has_detection and not has_yellow
       
        self.detection_history.append(is_valid)
        self.confidence_history.append(max_conf if is_valid else 0)
        self.color_history.append((has_red, has_blue, has_yellow))
       
        flash_count = self._count_flashes()
        is_flashing = flash_count >= self.flash_threshold
        is_police = is_flashing and (has_red or has_blue) and not has_yellow
       
        return {
            'has_detection': has_detection,
            'current_confidence': max_conf,
            'has_red': has_red,
            'has_blue': has_blue,
            'has_yellow': has_yellow,
            'flash_count': flash_count,
            'is_flashing': is_flashing,
            'is_police': is_police,
            'box_coords': box_coords
        }
   
    def _count_flashes(self):
        if len(self.detection_history) < 5:
            return 0
        flash_count = 0
        was_detected = False
        for detected in self.detection_history:
            if detected and not was_detected:
                flash_count += 1
            was_detected = detected
        return flash_count
   
    def reset(self):
        self.detection_history.clear()
        self.confidence_history.clear()
        self.color_history.clear()




# ============================================================
# STEP 8: TEST FUNCTION
# ============================================================




def analyze_audio(audio_path, audio_model, mean, std):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    duration = len(y) / sr
    chunk_samples = int(DURATION * sr)
    num_chunks = int(duration // DURATION)
   
    audio_results = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, len(y))
        chunk = y[start:end]
        start_sec = start / sr
        end_sec = end / sr
       
        if len(chunk) < sr * 0.5:
            continue
       
        if len(chunk) < TARGET_LENGTH:
            chunk = np.pad(chunk, (0, TARGET_LENGTH - len(chunk)))
        else:
            chunk = chunk[:TARGET_LENGTH]
       
        mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=N_MFCC)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        feat_input = np.expand_dims(mfccs_mean, axis=0)
        feat_input = (feat_input - mean) / (std + 1e-6)
       
        conf = audio_model.predict(feat_input, verbose=0)[0][0]
        is_siren = conf > 0.7
       
        audio_results.append({
            'start': start_sec,
            'end': end_sec,
            'confidence': float(conf),
            'is_siren': is_siren
        })
   
    return audio_results




def create_spectator_video(video_path, audio_results, vision_detector, output_path=None):
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"/content/spectator_{timestamp}.mp4"
   
    print(f"\n🎬 Creating spectator video...")
   
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   
    print(f"   Total frames: {total_frames}")
    print(f"   Video FPS: {fps}")
   
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
   
    frame_idx = 0
    ping_frame = 0
    vision_detections = 0
    processed_frames = 0
    vision_detector.reset()
   
    import sys
    next_progress = max(50, total_frames // 20)
   
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
       
        current_time = frame_idx / fps
       
        current_audio_conf = 0
        is_audio_siren = False
        for ar in audio_results:
            if ar['start'] <= current_time <= ar['end']:
                current_audio_conf = ar['confidence']
                is_audio_siren = ar['is_siren']
                break
       
        vision_result = vision_detector.analyze_frame(frame, frame_idx, fps)
       
        processed_frames += 1
        if vision_result['has_detection']:
            vision_detections += 1
       
        if frame_idx >= next_progress:
            percent = (frame_idx / total_frames) * 100
            sys.stdout.write(f"\r   Progress: {percent:.0f}% ({frame_idx}/{total_frames} frames)")
            sys.stdout.flush()
            next_progress += max(50, total_frames // 20)
       
        if vision_result['box_coords'] is not None and vision_result['is_police']:
            x1, y1, x2, y2 = map(int, vision_result['box_coords'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            badge = f"POLICE: {vision_result['current_confidence']:.0%}"
            cv2.rectangle(frame, (x1, y1-25), (x1 + 120, y1), (0, 0, 255), -1)
            cv2.putText(frame, badge, (x1 + 5, y1-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
       
        if is_audio_siren or vision_result['is_police']:
            if ping_frame % 10 < 5:
                red_overlay = np.full_like(frame, (0, 0, 50))
                frame = cv2.addWeighted(frame, 0.7, red_overlay, 0.3, 0)
       
        overlay = frame.copy()
        bar_height = 120
       
        if is_audio_siren or vision_result['is_police']:
            cv2.rectangle(overlay, (0, 0), (width, bar_height), (0, 0, 150), -1)
            status_text = "🚨 EMERGENCY VEHICLE DETECTED! 🚨"
            status_color = (0, 0, 255)
        else:
            cv2.rectangle(overlay, (0, 0), (width, bar_height), (0, 100, 0), -1)
            status_text = "✅ NO EMERGENCY DETECTED"
            status_color = (0, 255, 0)
       
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        cv2.putText(overlay, status_text, (width//2 - text_size[0]//2, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
       
        audio_color = (0, 255, 255) if current_audio_conf > 0.7 else (200, 200, 200)
        cv2.putText(overlay, f"🎤 AUDIO: {current_audio_conf:.1%}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, audio_color, 2)
       
        vision_color = (0, 255, 255) if vision_result['is_police'] else (200, 200, 200)
        cv2.putText(overlay, f"👁️ VISION: {vision_result['current_confidence']:.1%}", (20, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, vision_color, 2)
       
        cv2.putText(overlay, f"✨ FLASH COUNT: {vision_result['flash_count']}", (width - 300, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
       
        cv2.putText(overlay, f"⏱️ TIME: {current_time:.1f}s", (width - 200, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
       
        bar_y = height - 60
        meter_height = 25
       
        audio_width = int(current_audio_conf * 400)
        cv2.rectangle(overlay, (50, bar_y), (450, bar_y + meter_height), (50, 50, 50), -1)
        cv2.rectangle(overlay, (50, bar_y), (50 + audio_width, bar_y + meter_height), (0, 255, 255), -1)
        cv2.putText(overlay, f"AUDIO: {current_audio_conf:.1%}", (460, bar_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
       
        vision_width = int(vision_result['current_confidence'] * 400)
        cv2.rectangle(overlay, (width - 450, bar_y), (width - 50, bar_y + meter_height), (50, 50, 50), -1)
        cv2.rectangle(overlay, (width - 450, bar_y), (width - 450 + vision_width, bar_y + meter_height), (0, 255, 255), -1)
        cv2.putText(overlay, f"VISION: {vision_result['current_confidence']:.1%}", (width - 460, bar_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
       
        cv2.line(overlay, (50 + 280, bar_y), (50 + 280, bar_y + meter_height), (255, 255, 255), 1)
        cv2.line(overlay, (width - 450 + 280, bar_y), (width - 450 + 280, bar_y + meter_height), (255, 255, 255), 1)
       
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        out.write(result)
       
        frame_idx += 1
        ping_frame += 1
   
    print()
    cap.release()
    out.release()
   
    print(f"✅ Spectator video saved: {output_path}")
    return output_path, vision_detections, processed_frames




def test_video(video_path, audio_model, vision_model, mean, std):
    print("\n" + "="*70)
    print(f"📹 TESTING: {os.path.basename(video_path)}")
    print("="*70)


    audio_path = video_path.replace('.mp4', '_extracted.wav')
    process_with_ffmpeg(video_path, audio_path)
    print(f"   Audio extracted with FFmpeg (same as training)")


    audio_results = analyze_audio(audio_path, audio_model, mean, std)


    print("\n🔊 AUDIO RESULTS:")
    print("-" * 50)
    siren_chunks = 0
    for ar in audio_results:
        status = "🔴 SIREN" if ar['is_siren'] else "🟢 CLEAR"
        print(f"   {ar['start']:.0f}-{ar['end']:.0f}s: {ar['confidence']:.1%} | {status}")
        if ar['is_siren']:
            siren_chunks += 1


    audio_score = (siren_chunks / len(audio_results) * 100) if audio_results else 0
    print(f"\n📊 AUDIO SCORE: {siren_chunks}/{len(audio_results)} ({audio_score:.1f}%)")


    vision_detector = TemporalPoliceDetector(vision_model, history_frames=30, flash_threshold=3)
    output_path, vision_detections, processed_frames = create_spectator_video(
        video_path, audio_results, vision_detector
    )


    vision_score = (vision_detections / processed_frames * 100) if processed_frames > 0 else 0


    # FIXED: Simple average fusion - NO THRESHOLD ZEROING!
    # Both scores always contribute, even if one is 0%
    fused_score = (audio_score + vision_score) / 200


    if fused_score >= 0.45:
        verdict = "🚨 EMERGENCY VEHICLE DETECTED"
    else:
        verdict = "✅ NO THREAT"


    print(f"\n📊 VISION SCORE:  {vision_detections}/{processed_frames} ({vision_score:.1f}%)")
    print(f"🎤 AUDIO SCORE:   {siren_chunks}/{len(audio_results)} ({audio_score:.1f}%)")
    print(f"🔀 FUSION SCORE:  {fused_score:.1%}")
    print(f"🚨 FINAL VERDICT: {verdict}")


    print("\n🎬 SPECTATOR VIDEO:")
    display(Video(output_path, width=800, embed=True))


    print("\n🔊 Audio preview:")
    display(Audio(audio_path, rate=SAMPLE_RATE))


    return output_path




# ============================================================
# STEP 9: RUN THE SYSTEM
# ============================================================


mean = np.load("/content/train_mean.npy")
std = np.load("/content/train_std.npy")
print(f"\n✅ Loaded training mean and std (shape: {mean.shape})")


print("\n" + "="*70)
print("📤 UPLOAD YOUR TEST VIDEO")
print("="*70)
print("\n✅ Audio model trained!")
print(f"   - Positives: {total_pos} (all formats)")
print(f"   - Negatives: {total_neg} (all formats)")
print(f"   - Test accuracy: {test_acc:.2f}%")


if vision_model:
    print("\n✅ Vision model loaded (temporal detection + color filtering)")
else:
    print("\n⚠️ Audio-only mode (no vision detection)")


print("\n📁 Please upload your MP4 test video:")


uploaded = files.upload()


for video_path in uploaded.keys():
    output = test_video(video_path, audio_model, vision_model, mean, std)
    print(f"\n✅ Done! Spectator video saved: {output}")


print("\n" + "="*70)
print("✅ COMPLETE!")
print("="*70)
print("\n🔑 FIXED: Now includes FLAC files in training!")
print(f"   Total negative samples: {total_neg} (includes {neg_flac} FLAC files)")
