import os
import glob
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from zhou_method_revisi import process_signal, detect_peaks_valleys, compute_E_peak_valley, estimate_bp


subject_path = os.path.join(os.getcwd(), "Dataset Pyhsio ITERA", "ades2", "rgb")
GT_PATH = os.path.join(os.getcwd(), "01-01", "O1-O1.json")
bmi = 19.59

def butter_lowpass_filter(data, cutoff=2.0, fs=30.0, order=2):
    """
    Filter low-pass untuk menghilangkan noise tinggi dari sinyal.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def get_face_mean_rgb(subject_path):
    folder_name = os.path.basename(subject_path)
    output_folder = os.path.join(subject_path, "output_roi")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(f"\n📌 Memproses folder: {folder_name}")
    image_paths = sorted(glob.glob(os.path.join(subject_path, "*.jpg")))
    
    if not image_paths:
        raise ValueError(f"❌ Tidak ditemukan gambar di {subject_path}")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    
    roi_rgb_values = {roi: {'R': [], 'G': [], 'B': []} for roi in get_roi_definitions().keys()}
    
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠ Tidak dapat membaca gambar {image_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = image.shape
            roi_definitions = get_roi_definitions()
            
            for roi_name, (indices, color) in roi_definitions.items():
                x1, y1, x2, y2 = extract_roi_coordinates(face_landmarks.landmark, indices, w, h)
                roi = image_rgb[y1:y2, x1:x2]
                
                if roi.size > 0:
                    mean_rgb = np.mean(roi, axis=(0, 1))
                    roi_rgb_values[roi_name]['R'].append(mean_rgb[0])
                    roi_rgb_values[roi_name]['G'].append(mean_rgb[1])
                    roi_rgb_values[roi_name]['B'].append(mean_rgb[2])
                else:
                    print(f"⚠ Region kosong di {roi_name} pada {image_path}")
    
    face_mesh.close()
    return roi_rgb_values

def main():
    mean_rgb_signal = get_face_mean_rgb(subject_path)
    
    for roi_name, rgb_values in mean_rgb_signal.items():
        print(f"\n🔍 Memproses ROI: {roi_name}")
        V_RGB = np.array([rgb_values['R'], rgb_values['G'], rgb_values['B']])
        
        if V_RGB.shape[1] < 10:
            print(f"⚠ Data untuk ROI {roi_name} terlalu sedikit. Melewati...")
            continue
        
        S = process_signal(V_RGB)
        S_filtered = butter_lowpass_filter(S)
        peaks, valleys = detect_peaks_valleys(S_filtered)
        
        if len(peaks) < 2 or len(valleys) < 2:
            print(f"⚠ Tidak cukup peaks/valleys untuk ROI {roi_name}, hasil mungkin tidak valid.")
            continue
        
        E_peak, E_valley = compute_E_peak_valley(S_filtered, peaks, valleys)
        
        if E_peak is None or E_valley is None:
            print(f"⚠ Tidak dapat menghitung E_peak/E_valley untuk ROI {roi_name}. Melewati...")
            continue
        
        SBP, DBP = estimate_bp(E_peak, E_valley, bmi)
        print(f"🩺 ROI {roi_name} - SBP: {SBP:.1f} mmHg | DBP: {DBP:.1f} mmHg")
        
        output_file = os.path.join(subject_path, f"{roi_name}_bp_results.txt")
        with open(output_file, "w") as f:
            f.write(f"ROI: {roi_name}\n")
            f.write(f"SBP: {SBP:.1f} mmHg\n")
            f.write(f"DBP: {DBP:.1f} mmHg\n")
        print(f"📂 Hasil SBP & DBP {roi_name} disimpan di {output_file}")

if __name__ == "__main__":
    main()
