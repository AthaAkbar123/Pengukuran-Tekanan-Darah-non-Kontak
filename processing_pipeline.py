from zhou_method_revisi import process_signal, detect_peaks_valleys, compute_E_peak_valley, estimate_bp
import os
import glob
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

subject_path = os.path.join(os.getcwd(), "Dataset", "adin2", "rgb")
GT_PATH = os.path.join(os.getcwd(), "01-01", "O1-O1.json")

bmi = 24

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
    Memproses semua gambar PNG dalam path yang diberikan, mendeteksi wajah menggunakan MediaPipe,
    dan menghitung nilai rata-rata RGB dari region wajah.
    
    Args:
        subject_path (str): Path ke direktori yang berisi gambar PNG
        
    Returns:
        np.ndarray: Array sinyal rata-rata RGB dengan bentuk (3, num_frames)
    """
    # Mendapatkan nama folder untuk file .npy
    folder_name = os.path.basename(subject_path)
    npy_file_path = os.path.join(subject_path, f"{folder_name}.npy")
    
    # Memeriksa apakah file .npy sudah ada
    os.path.exists(npy_file_path)

    if os.path.exists(npy_file_path):
        print(f"Memuat nilai rata-rata RGB yang sudah ada dari {npy_file_path}")
        mean_rgb_values = np.load(npy_file_path)
    else:
        print(f"Data tersimpan tidak ditemukan. Memproses gambar dari {subject_path}")
        # Mendapatkan semua gambar PNG dalam direktori
        image_paths = sorted(glob.glob(os.path.join(subject_path, "*.jpg")))
        
        if not image_paths:
            raise ValueError(f"Tidak ditemukan gambar PNG di {subject_path}")
        
        # Inisialisasi deteksi wajah MediaPipe
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        # Inisialisasi array untuk menyimpan nilai rata-rata RGB
        mean_rgb_values = np.zeros((3, len(image_paths)))
        
        # Memproses setiap gambar
        for i, image_path in enumerate(image_paths):
            # Membaca gambar
            image = cv2.imread(image_path)
            # Mencetak status kemajuan
            print(f"Memproses gambar {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            if image is None:
                print(f"Peringatan: Tidak dapat membaca gambar {image_path}")
                continue
            
            # Mengkonversi ke RGB untuk MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Mendeteksi wajah
            results = face_detection.process(image_rgb)
            
            if results.detections:
                # Mendapatkan bounding box dari wajah pertama yang terdeteksi
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                
                # Mendapatkan dimensi gambar
                h, w, _ = image.shape
                
                # Menghitung koordinat absolut
                xmin = int(bboxC.xmin * w)
                ymin = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Memastikan koordinat berada dalam batas gambar
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                width = min(width, w - xmin)
                height = min(height, h - ymin)
                
                # Mengekstrak region wajah
                face_region = image_rgb[ymin:ymin+height, xmin:xmin+width]
                
                if face_region.size > 0:
                    # Menghitung nilai rata-rata RGB untuk region wajah
                    mean_rgb = np.mean(face_region, axis=(0, 1))
                    mean_rgb_values[:, i] = mean_rgb
                else:
                    print(f"Peringatan: Region wajah kosong pada gambar {image_path}")
            else:
                print(f"Peringatan: Tidak ada wajah terdeteksi pada gambar {image_path}")
        
        # Menutup objek deteksi wajah MediaPipe
        face_detection.close()
        
        # Menyimpan nilai rata-rata RGB yang diekstrak
        print(f"Menyimpan nilai rata-rata RGB ke {npy_file_path}")
        np.save(npy_file_path, mean_rgb_values)
    
    # Membuat plot sinyal RGB
    plt.figure(figsize=(12, 6))
    time = np.arange(mean_rgb_values.shape[1])
    
    plt.plot(time, mean_rgb_values[0], 'r-', label='Kanal merah')
    plt.plot(time, mean_rgb_values[1], 'g-', label='Kanal hijau')
    plt.plot(time, mean_rgb_values[2], 'b-', label='Kanal biru')
    
    plt.title(f'Sinyal Rata-rata RGB dari Region Wajah - {folder_name}')
    plt.xlabel('Nomor Frame')
    plt.ylabel('Intensitas')
    plt.legend()
    plt.grid(True)
    
    # Menyimpan plot ke direktori level pertama (di samping file JSON)
    parent_dir = os.path.dirname(subject_path)
    plot_path = os.path.join(parent_dir, f"{folder_name}_rgb_signals.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot sinyal RGB disimpan ke {plot_path}")
    
    # Menutup plot untuk membebaskan memori
    plt.close()
    
    return mean_rgb_values

def main():
    # 1. Mendapatkan nilai rata-rata RGB dari region wajah
    mean_rgb_signal = get_face_mean_rgb(subject_path)
    
    # 2. Memproses sinyal rata-rata RGB (Versi Asli Zhou et al.)
    S = process_signal(mean_rgb_signal)
    
    # 2. Bisa juga pakai versi alternatif, dimana dari mean_rgb_signal langsung dikirimkan ke fungsi POS lalu lanjut ke step 3
    
    # 3. Mendeteksi puncak dan lembah
    peaks, valleys = detect_peaks_valleys(S)
    print(f"Jumlah puncak: {len(peaks)} | Jumlah lembah: {len(valleys)}")
    
    # 4. Menghitung E_peak dan E_valley
    E_peak, E_valley = compute_E_peak_valley(S, peaks, valleys)
    print(f"E_peak: {E_peak} | E_valley: {E_valley}")

    # 5. Print SBP dan DBP
    SBP, DBP = estimate_bp(E_peak, E_valley, bmi)
    print(f"SBP: {SBP:.1f} mmHg | DBP: {DBP:.1f} mmHg")

if __name__ == "__main__":
    main()