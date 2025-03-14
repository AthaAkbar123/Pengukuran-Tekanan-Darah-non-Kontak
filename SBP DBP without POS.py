import pandas as pd
from zhou_method_revisi import process_signal, detect_peaks_valleys, compute_E_peak_valley, estimate_bp
import os
import glob
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt


subject_path = os.path.join(os.getcwd(), "Dataset", "adin2", "rgb")
GT_PATH = os.path.join(os.getcwd(), "01-01", "O1-O1.json")

# Path ke file Excel yang berisi data berat dan tinggi badan
BMI_FILE = os.path.join(os.getcwd(), "Dataset", "Data Master.xlsx")

# Membaca file Excel dan menghitung BMI
df_bmi = pd.read_excel(BMI_FILE)

# Menambahkan kolom BMI jika belum ada
if "BMI" not in df_bmi.columns:
    df_bmi["BMI"] = df_bmi["WEIGHT"] / ((df_bmi["HEIGHT"] ** 2) )

# Membuat dictionary dengan format {nama_folder: BMI}
bmi_data = {str(row["Nama Folder"]): row["BMI"] for _, row in df_bmi.iterrows()}

def get_roi_definitions():
    return {
        'Jidat': ([54, 63, 109, 107, 338, 336, 293], (255, 0, 0)),
        'PipiKanan': ([34, 230, 203, 206, 207, 213], (0, 255, 0)),
        'PipiKiri': ([264, 450, 423, 426, 427, 433], (0, 0, 255)),
        'Hidung': ([8, 193, 417, 48, 278, 4, 44, 274], (255, 255, 0)),
        'Dagu': ([210, 430, 150, 379, 152], (255, 0, 255)),
    }

def extract_roi_coordinates(landmarks, indices, image_width, image_height): 
    points = [landmarks[idx] for idx in indices]
    coords = [(int(point.x * image_width), int(point.y * image_height)) for point in points]
    min_x, max_x = min(pt[0] for pt in coords), max(pt[0] for pt in coords)
    min_y, max_y = min(pt[1] for pt in coords), max(pt[1] for pt in coords)
    return max(0, min_x), max(0, min_y), min(image_width, max_x), min(image_height, max_y)

def get_face_mean_rgb(subject_path):
    """
    Memproses semua gambar dalam path yang diberikan, mendeteksi wajah menggunakan MediaPipe,
    menghitung nilai rata-rata RGB dari setiap ROI yang telah didefinisikan,
    dan mengolah sinyal menjadi SBP & DBP.
    """
    folder_name = os.path.basename(subject_path)
    output_folder = os.path.join(subject_path, "output_roi")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"\n📌 Memproses folder: {folder_name}")

    # Ambil nilai BMI dari file Excel berdasarkan nama folder
    bmi = bmi_data.get(folder_name, 24)  # Default ke 24 jika tidak ditemukan di file Excel

    image_paths = sorted(glob.glob(os.path.join(subject_path, "*.jpg")))

    if not image_paths:
        raise ValueError(f"❌ Tidak ditemukan gambar di {subject_path}")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    roi_rgb_values = {roi: {'R': [], 'G': [], 'B': []} for roi in get_roi_definitions().keys()}

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        print(f"  ▶ Memproses gambar {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        if image is None:
            print(f"  ⚠ Peringatan: Tidak dapat membaca gambar {image_path}")
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
                    print(f"  ⚠ Region kosong di {roi_name} pada {image_path}")

    face_mesh.close()

    for roi_name in roi_rgb_values:
        npy_path = os.path.join(output_folder, f"{roi_name}_rgb.npy")
        np.save(npy_path, roi_rgb_values[roi_name])
        print(f"  ✅ Data RGB {roi_name} disimpan di {npy_path}")

    return roi_rgb_values, bmi

def main():
    subject_path = os.path.join(os.getcwd(), "Dataset", "adin2", "rgb")
    
    mean_rgb_signal, bmi = get_face_mean_rgb(subject_path)

    for roi_name, rgb_values in mean_rgb_signal.items():
        print(f"\n🔍 Memproses ROI: {roi_name}")
        
        V_RGB = np.array([rgb_values['R'], rgb_values['G'], rgb_values['B']])

        S = process_signal(V_RGB)

        peaks, valleys = detect_peaks_valleys(S)
        print(f"📊 ROI {roi_name} - Puncak: {len(peaks)} | Lembah: {len(valleys)}")

        E_peak, E_valley = compute_E_peak_valley(S, peaks, valleys)
        print(f"📈 ROI {roi_name} - E_peak: {E_peak} | E_valley: {E_valley}")

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
