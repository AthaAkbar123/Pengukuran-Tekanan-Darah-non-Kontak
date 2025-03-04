import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy.signal import butter, filtfilt, find_peaks

# Constants
IMAGE_FOLDER_PATH = os.path.join(os.getcwd(), 'PURE Dataset', '01-01', '01-01')
FPS = 30  # Sesuaikan dengan FPS video (simulasi)
BMI = 24  # Nilai BMI yang diberikan

# Define ROIs
def get_roi_definitions():
    return {
        'Jidat': ([54, 63, 109, 107, 338, 336, 293], (255, 0, 0)),
        'PipiKanan': ([34, 230, 203, 206, 207, 213], (0, 255, 0)),
        'PipiKiri': ([264, 450, 423, 426, 427, 433], (0, 0, 255)),
        'Hidung': ([8, 193, 417, 48, 278, 4, 44, 274], (255, 255, 0)),
        'Dagu': ([210, 430, 150, 379, 152], (255, 0, 255)),
    }

# Bandpass filter function
def bandpass_filter(signal, lowcut=0.5, highcut=2.0, fs=FPS, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Extract ROI coordinates
def extract_roi_coordinates(landmarks, indices, image_width, image_height):
    points = [landmarks[idx] for idx in indices]
    coords = [(int(point.x * image_width), int(point.y * image_height)) for point in points]
    min_x, max_x = min(pt[0] for pt in coords), max(pt[0] for pt in coords)
    min_y, max_y = min(pt[1] for pt in coords), max(pt[1] for pt in coords)
    return max(0, min_x), max(0, min_y), min(image_width, max_x), min(image_height, max_y)

# Compute rPPG signal
def compute_rppg_signal(r_signal, g_signal, b_signal):
    return (r_signal + g_signal + b_signal) / 3

# Calculate SBP & DBP
def estimate_bp(E_peak, E_valley, bmi):
    SBP = 23.7889 + 95.4335 * E_peak + 4.5958 * bmi - 5.109 * E_peak * bmi
    DBP = -17.3772 - 115.1747 * E_valley + 4.0251 * bmi + 5.2825 * E_valley * bmi
    return SBP, DBP

# Main function
def main():
    image_files = sorted(glob(os.path.join(IMAGE_FOLDER_PATH, '*.png')))
    
    if len(image_files) == 0:
        print("Error: No images found in the folder.")
        return
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    
    rgb_signals_per_roi = {roi: {'R': [], 'G': [], 'B': []} for roi in get_roi_definitions().keys()}
    
    for image_path in image_files:
        frame = cv2.imread(image_path)
        if frame is None:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                roi_definitions = get_roi_definitions()
                for roi_name, (indices, color) in roi_definitions.items():
                    x1, y1, x2, y2 = extract_roi_coordinates(face_landmarks.landmark, indices, w, h)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        rgb_signals_per_roi[roi_name]['R'].append(np.mean(roi[:, :, 0]))
                        rgb_signals_per_roi[roi_name]['G'].append(np.mean(roi[:, :, 1]))
                        rgb_signals_per_roi[roi_name]['B'].append(np.mean(roi[:, :, 2]))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    for roi_name in rgb_signals_per_roi:
        print(f"Processing ROI: {roi_name}")
        r_signal = np.array(rgb_signals_per_roi[roi_name]['R'])
        g_signal = np.array(rgb_signals_per_roi[roi_name]['G'])
        b_signal = np.array(rgb_signals_per_roi[roi_name]['B'])

        if len(r_signal) > 10 and len(g_signal) > 10 and len(b_signal) > 10:
            rppg_signal = compute_rppg_signal(r_signal, g_signal, b_signal)
            filtered_rppg_signal = bandpass_filter(rppg_signal)

            # Find peaks and valleys
            peaks, _ = find_peaks(filtered_rppg_signal)
            valleys, _ = find_peaks(-filtered_rppg_signal)
            
            if len(peaks) > 0 and len(valleys) > 0:
                E_peak = np.mean(filtered_rppg_signal[peaks])
                E_valley = np.mean(filtered_rppg_signal[valleys])
                SBP, DBP = estimate_bp(E_peak, E_valley, BMI)
                print(f"{roi_name} - Estimated BP: SBP = {SBP:.1f} mmHg, DBP = {DBP:.1f} mmHg")
            
            plt.figure(figsize=(15, 5))
            plt.plot(filtered_rppg_signal, label="Filtered rPPG Signal", color='black', linewidth=2)
            plt.scatter(peaks, filtered_rppg_signal[peaks], color='red', label="Peaks", marker='o')
            plt.scatter(valleys, filtered_rppg_signal[valleys], color='blue', label="Valleys", marker='x')
            plt.title(f"{roi_name} - Filtered rPPG with Peaks and Valleys")
            plt.xlabel("Frames")
            plt.ylabel("Intensity")
            plt.legend()
            plt.grid(True)
            plt.show()

if __name__ == '__main__':
    main()
