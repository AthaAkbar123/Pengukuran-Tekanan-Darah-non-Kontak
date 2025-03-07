import os
import random

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, filtfilt, find_peaks
from sklearn.decomposition import FastICA

from utils import count_percentage

seed_number = 2023
random.seed(seed_number)
np.random.seed(seed_number)

def load_subject_list(dataset_root):
    sub_list = [name for name in os.listdir(dataset_root) if name.endswith('6')]
    sub_list.sort()
    
        
    return sub_list

def rppg_ica(subject):
    v_rgb = np.load(os.path.join("data", subject, "mean_rgb_all.npy")).T
    v_rgb = detrend(v_rgb)
    lowcut = 10 / (v_rgb.shape[1] / 2)
    b, a = butter(1, lowcut, btype='high')
    v_rgb = np.array([filtfilt(b, a, row) for row in v_rgb])
    
    mu = np.mean(v_rgb, axis=1, keepdims = True)
    sigma = np.std(v_rgb, axis=1, keepdims = True)
    x_t = (v_rgb - mu) / sigma
    
    ica = FastICA(n_components=3, random_state=seed_number, whiten='unit-variance')
    idp_com_ica = ica.fit_transform(x_t.T).T
    
    return idp_com_ica[0, :]

def bandpass(signal, bandpass=[0.7, 2.2], fs=30):
    b, a = butter(3, bandpass, btype='bandpass', fs=fs)
    return filtfilt(b, a, signal)

def findpeaks(signal):
    peak_ind, _ = find_peaks(signal)
    valley_ind, _ = find_peaks(-signal)
    
    return peak_ind, valley_ind

def get_e_value(signal, peak_ind, valley_ind):
    h_d = signal[peak_ind]
    h_l = signal[valley_ind]
    n_1 = len(h_d)
    n_2 = len(h_l)
    e_peak = np.sum(h_d) / n_1 if n_1 != 0 else 0
    e_valley = np.sum(h_l) / n_2 if n_2 != 0 else 0
    return e_peak, e_valley

def get_data_csv(df, subject_name):
    sub_name = subject_name[:-1]
    
    # find sub_name on df on which row
    row_idx = df.index[df['ALLIAS'] == sub_name].tolist()[0]
    
    data_csv = {
        "weight" : df.at[row_idx, 'WEIGHT'],
        "height" : df.at[row_idx, 'HEIGHT'],
        "sys" : df.at[row_idx, 'SYSTOLIC'],
        "dia" : df.at[row_idx, 'DIASTOLIC'],
        "bmi" : df.at[row_idx, 'WEIGHT'] / (df.at[row_idx, 'HEIGHT'] / 100) ** 2,
    }
    
    return data_csv

def zhou_methods(e_peak, e_valley, bmi):
    est_sbp = 23.7889 + 95.4335 * e_peak + 4.5958 * bmi - 5.109 * e_peak * bmi
    est_dbp = -17.3772 - 115.1747 * e_valley + 4.0251 * bmi + 5.2825 * e_valley * bmi
    
    return est_sbp, est_dbp
    


if __name__ == "__main__":
    
    sbp_error_list = []
    dbp_error_list = []
    
    # 1. Load subject list and Dataframe
    sub_list = load_subject_list(
        dataset_root="data"
    )
    
    df = pd.read_csv("data/data_collection_record.csv")
    df = df.drop(df.columns[0], axis=1)
    
    # 2. Iterate over subject list
    for i, subject in enumerate(sub_list):
        print(f"Processing subject {i+1}/{len(sub_list)}")
        
        # 3. Load subject's rPPG signal
        rppg = rppg_ica(subject)
        
        # 4. Bandpass filter
        rppg = bandpass(rppg)
        
        # 5. Find peaks and valleys
        peak_ind, valley_ind = findpeaks(rppg)
        
        # 6. Get e_peak and e_valley
        e_peak, e_valley = get_e_value(rppg, peak_ind, valley_ind)
        
        # 7. Get data from master csv
        data_csv = get_data_csv(
            df=df,
            subject_name=subject
        )
        
        # 8. Calculate the estimated SBP and DBP
        est_sbp, est_db = zhou_methods(
            e_peak=e_peak,
            e_valley=e_valley,
            bmi=data_csv["bmi"]
        )
        
        # 9. Compare with ground truth
        sbp_err = abs(est_sbp - data_csv["sys"])
        dbp_err = abs(est_db - data_csv["dia"])
        
        print(f"SBP: {sbp_err}, DBP: {dbp_err}")
        
        # 10. Append to list
        sbp_error_list.append(sbp_err)
        dbp_error_list.append(dbp_err)
        
    # 11. Count the MAE, RMSE, SD, CE5, CE10, CE15
    sbp_mae = np.mean(sbp_error_list)
    sbp_rmse = np.sqrt(np.mean(np.square(sbp_error_list)))
    sbp_sd = np.std(sbp_error_list)
    sbp_ce5, sbp_ce10, sbp_ce15 = count_percentage(sbp_error_list)
    
    dbp_mae = np.mean(dbp_error_list)
    dbp_rmse = np.sqrt(np.mean(np.square(dbp_error_list)))
    dbp_sd = np.std(dbp_error_list)
    dbp_ce5, dbp_ce10, dbp_ce15 = count_percentage(dbp_error_list)
    
    print(f"Systolic -> MAE: {sbp_mae}, RMSE: {sbp_rmse}, SD: {sbp_sd}, CE5: {sbp_ce5}, CE10: {sbp_ce10}, CE15: {sbp_ce15}")
    print(f"Diastolic -> MAE: {dbp_mae}, RMSE: {dbp_rmse}, SD: {dbp_sd}, CE5: {dbp_ce5}, CE10: {dbp_ce10}, CE15: {dbp_ce15}")

    