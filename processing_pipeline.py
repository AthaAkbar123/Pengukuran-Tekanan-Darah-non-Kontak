from ZhouMethod import process_signal, detect_peaks_valleys, compute_E_peak_valley, estimate_bp
import os
import glob
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import re
import csv
import pandas as pd
import datetime

# Remove hardcoded subject path
dataset_path = os.path.join(os.getcwd(), "Dataset")
GT_PATH = os.path.join(os.getcwd(), "01-01", "O1-O1.json")

# Path to the CSV file containing BMI values - ensure it points to the Data Master.csv file
bmi_csv_path = os.path.join(os.getcwd(), "Dataset", "Data Master.csv")

def get_unique_filename(base_path, use_timestamp=True):

    base_dir = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    name, ext = os.path.splitext(base_name)
    
    if use_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(base_dir, f"{name}_{timestamp}{ext}")
    else:
        counter = 1
        new_path = base_path
        while os.path.exists(new_path):
            new_path = os.path.join(base_dir, f"{name}_{counter}{ext}")
            counter += 1
        return new_path

def load_bmi_data(csv_path):

    bmi_data = {}
    
    try:
        print(f"Loading BMI data from {csv_path}")
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            
            print(f"CSV columns found: {df.columns.tolist()}")
            
            if 'ALLIAS' in df.columns and 'BMI' in df.columns:
                for _, row in df.iterrows():
                    try:
                        if pd.notna(row['ALLIAS']):
                            subject_name = str(row['ALLIAS']).strip()
                            if pd.notna(row['BMI']) and subject_name:
                                bmi_value = float(row['BMI'])
                                bmi_data[subject_name] = bmi_value
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Error processing row: {e}")
                        continue
            else:
                print(f"Warning: CSV must have 'ALLIAS' and 'BMI' columns. Found columns: {df.columns.tolist()}")
        except ImportError:
            print("Pandas not available, using built-in CSV reader")
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if 'ALLIAS' in row and 'BMI' in row:
                            subject_name = str(row['ALLIAS']).strip()
                            if subject_name and row['BMI']:
                                bmi_value = float(row['BMI'])
                                if subject_name and bmi_value > 0:  
                                    bmi_data[subject_name] = bmi_value
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Error processing row: {e}")
                        continue
        
        normalized_bmi_data = {}
        for key, value in bmi_data.items():
            normalized_key = key.strip().lower()
            normalized_bmi_data[normalized_key] = value
                        
        print(f"Loaded BMI data for {len(normalized_bmi_data)} subjects")
        
        if normalized_bmi_data:
            print("Sample BMI data (first 5 entries):")
            for i, (subject, bmi) in enumerate(list(normalized_bmi_data.items())[:5]):
                print(f"  {subject}: {bmi}")
                
        return normalized_bmi_data
        
    except Exception as e:
        print(f"Error loading BMI data from {csv_path}: {str(e)}")
        return {}

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

def natural_sort_key(text):
    return [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', text)]

def get_subject_folders(dataset_path):
    subject_folders = []
    for subject_folder in os.listdir(dataset_path):
        subject_dir = os.path.join(dataset_path, subject_folder)
        if os.path.isdir(subject_dir):
            rgb_dir = os.path.join(subject_dir, "rgb")
            if os.path.exists(rgb_dir):
                subject_folders.append(rgb_dir)
    
    return subject_folders

def get_roi_mean_rgb(subject_path, roi_name=None):

    folder_name = os.path.basename(os.path.dirname(subject_path))
    
    file_suffix = f"_{roi_name}" if roi_name else ""
    npy_file_path = os.path.join(subject_path, f"{folder_name}{file_suffix}.npy")
    
    if os.path.exists(npy_file_path):
        print(f"Loading existing RGB averages from {npy_file_path}")
        mean_rgb_values = np.load(npy_file_path)
    else:
        print(f"No cached data found. Processing images from {subject_path} for {roi_name or 'full face'}")
        image_paths = sorted(glob.glob(os.path.join(subject_path, "*.jpg")), key=natural_sort_key)
        
        if not image_paths:
            raise ValueError(f"No images found in {subject_path}")
        
        if roi_name and roi_name in get_roi_definitions():
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, 
                                              max_num_faces=1,
                                              min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5)
            roi_definitions = get_roi_definitions()
            roi_indices, roi_color = roi_definitions[roi_name]
        else:
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        mean_rgb_values = np.zeros((3, len(image_paths)))
        
        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)} for {roi_name or 'full face'}")
            if image is None:
                print(f"Warning: Unable to read image {image_path}")
                continue    
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if roi_name and roi_name in get_roi_definitions():
                results = face_mesh.process(image_rgb)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    h, w, _ = image.shape
                    
                    x_min, y_min, x_max, y_max = extract_roi_coordinates(landmarks, roi_indices, w, h)
                    
                    roi_region = image_rgb[y_min:y_max, x_min:x_max]
                    
                    if roi_region.size > 0:
                        mean_rgb = np.mean(roi_region, axis=(0, 1))
                        mean_rgb_values[:, i] = mean_rgb
                    else:
                        print(f"Warning: Empty ROI region for {roi_name} in image {image_path}")
                else:
                    print(f"Warning: No face landmarks detected in image {image_path}")
            else:
                results = face_detection.process(image_rgb)
                
                if results.detections:
                    detection = results.detections[0]
                    bboxC = detection.location_data.relative_bounding_box
                    
                    h, w, _ = image.shape
                    
                    xmin = int(bboxC.xmin * w)
                    ymin = int(bboxC.ymin * h)
                    width = int(bboxC.width * w)
                    height = int(bboxC.height * h)
                    
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    width = min(width, w - xmin)
                    height = min(height, h - ymin)
                    
                    face_region = image_rgb[ymin:ymin+height, xmin:xmin+width]
                    
                    if face_region.size > 0:
                        mean_rgb = np.mean(face_region, axis=(0, 1))
                        mean_rgb_values[:, i] = mean_rgb
                    else:
                        print(f"Warning: Empty face region in image {image_path}")
                else:
                    print(f"Warning: No face detected in image {image_path}")
        
        if roi_name and roi_name in get_roi_definitions():
            face_mesh.close()
        else:
            face_detection.close()
        
        print(f"Saving RGB averages to {npy_file_path}")
        np.save(npy_file_path, mean_rgb_values)
    
    plt.figure(figsize=(12, 6))
    time = np.arange(mean_rgb_values.shape[1])
    
    plt.plot(time, mean_rgb_values[0], 'r-', label='Red channel')
    plt.plot(time, mean_rgb_values[1], 'g-', label='Green channel')
    plt.plot(time, mean_rgb_values[2], 'b-', label='Blue channel')
    
    plot_title = f'Average RGB Signal from {roi_name or "Face"} Region - {folder_name}'
    plt.title(plot_title)
    plt.xlabel('Frame Number')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    
    parent_dir = os.path.dirname(subject_path)
    plot_suffix = f"_{roi_name}" if roi_name else ""
    plot_path = os.path.join(parent_dir, f"{folder_name}{plot_suffix}_rgb_signals.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"RGB signal plot saved to {plot_path}")
    
    plt.close()
    
    return mean_rgb_values

def get_face_mean_rgb(subject_path):
    return get_roi_mean_rgb(subject_path)

def process_subject(subject_path, bmi):
    subject_name = os.path.basename(os.path.dirname(subject_path))
    print(f"\n\nProcessing subject: {subject_name}")
    
    results = {}
    
    print("\nProcessing full face:")
    mean_rgb_signal = get_face_mean_rgb(subject_path)
    S = process_signal(mean_rgb_signal)
    peaks, valleys = detect_peaks_valleys(S)
    print(f"Number of peaks: {len(peaks)} | Number of valleys: {len(valleys)}")
    
    E_peak, E_valley = compute_E_peak_valley(S, peaks, valleys)
    print(f"E_peak: {E_peak} | E_valley: {E_valley}")

    SBP, DBP = estimate_bp(E_peak, E_valley, bmi)
    print(f"SBP: {SBP:.1f} mmHg | DBP: {DBP:.1f} mmHg")
    
    results["FullFace"] = {"SBP": SBP, "DBP": DBP}
    
    for roi_name in get_roi_definitions().keys():
        print(f"\nProcessing ROI: {roi_name}")
        try:
            roi_rgb_signal = get_roi_mean_rgb(subject_path, roi_name)
            
            S_roi = process_signal(roi_rgb_signal)
            
            peaks_roi, valleys_roi = detect_peaks_valleys(S_roi)
            print(f"Number of peaks: {len(peaks_roi)} | Number of valleys: {len(valleys_roi)}")
            
            if len(peaks_roi) == 0 or len(valleys_roi) == 0:
                print(f"Warning: No peaks/valleys detected for ROI {roi_name}. Skipping.")
                continue
                
            E_peak_roi, E_valley_roi = compute_E_peak_valley(S_roi, peaks_roi, valleys_roi)
            print(f"E_peak: {E_peak_roi} | E_valley: {E_valley_roi}")

            SBP_roi, DBP_roi = estimate_bp(E_peak_roi, E_valley_roi, bmi)
            print(f"SBP: {SBP_roi:.1f} mmHg | DBP: {DBP_roi:.1f} mmHg")
            
            results[roi_name] = {"SBP": SBP_roi, "DBP": DBP_roi}
            
        except Exception as e:
            print(f"Error processing ROI {roi_name}: {str(e)}")
    
    return results
    
def main():
    bmi_data = load_bmi_data(bmi_csv_path)
    if not bmi_data:
        print("No BMI data found or error loading BMI data. Exiting.")
        return
    
    subject_folders = get_subject_folders(dataset_path)
    
    if not subject_folders:
        print("No subject folders found in dataset path!")
        return
        
    print(f"Found {len(subject_folders)} subject folders to process.")
    
    results = {}
    
    for subject_path in subject_folders:
        subject_folder_name = os.path.basename(os.path.dirname(subject_path))
        
        subject_bmi = None
        if subject_folder_name in bmi_data:
            subject_bmi = bmi_data[subject_folder_name]
        else:
            subject_lower = subject_folder_name.lower()
            if subject_lower in bmi_data:
                subject_bmi = bmi_data[subject_lower]
            else:
                base_name = re.sub(r'\d+$', '', subject_lower)
                if base_name in bmi_data:
                    subject_bmi = bmi_data[base_name]
                else:
                    for bmi_key in bmi_data.keys():
                        if bmi_key.lower() == subject_lower or bmi_key.lower() in subject_lower or subject_lower in bmi_key.lower():
                            subject_bmi = bmi_data[bmi_key]
                            print(f"Found BMI match: '{subject_folder_name}' -> '{bmi_key}'")
                            break
        
        if subject_bmi is not None:
            print(f"Using BMI value {subject_bmi} for subject {subject_folder_name}")
        else:
            print(f"Warning: No BMI data found for subject {subject_folder_name}. Skipping this subject.")
            continue
        
        try:
            subject_results = process_subject(subject_path, subject_bmi)
            results[subject_folder_name] = {"BMI": subject_bmi, "ROIs": subject_results}
        except Exception as e:
            print(f"Error processing subject {subject_folder_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n\n==== SUMMARY OF RESULTS ====")
    for subject, data in results.items():
        print(f"\nSubject: {subject} (BMI: {data['BMI']:.1f})")
        for roi_name, bp_values in data['ROIs'].items():
            print(f"  {roi_name}: SBP: {bp_values['SBP']:.1f} mmHg | DBP: {bp_values['DBP']:.1f} mmHg")
    
    base_results_path = os.path.join(os.getcwd(), "bp_results_all_rois.csv")
    results_path = get_unique_filename(base_results_path)
    
    with open(results_path, 'w', newline='') as f:
        header = "Subject,BMI"
        roi_names = ["FullFace"] + list(get_roi_definitions().keys())
        for roi in roi_names:
            header += f",{roi}_SBP,{roi}_DBP"
        f.write(header + "\n")
        
        for subject, data in results.items():
            row = f"{subject},{data['BMI']:.1f}"
            for roi in roi_names:
                if roi in data['ROIs']:
                    row += f",{data['ROIs'][roi]['SBP']:.1f},{data['ROIs'][roi]['DBP']:.1f}"
                else:
                    row += ",," 
            f.write(row + "\n")
    
    print(f"\nDetailed results saved to {results_path}")
    
    base_orig_results_path = os.path.join(os.getcwd(), "bp_results.csv")
    orig_results_path = get_unique_filename(base_orig_results_path)
    
    with open(orig_results_path, 'w', newline='') as f:
        f.write("Subject,BMI,SBP,DBP\n")
        for subject, data in results.items():
            if "FullFace" in data['ROIs']:
                sbp = data['ROIs']["FullFace"]["SBP"]
                dbp = data['ROIs']["FullFace"]["DBP"]
                f.write(f"{subject},{data['BMI']:.1f},{sbp:.1f},{dbp:.1f}\n")
    
    print(f"Original format results saved to {orig_results_path}")

if __name__ == "__main__":
    main()
