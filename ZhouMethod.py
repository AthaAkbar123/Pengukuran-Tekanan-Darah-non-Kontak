import numpy as np
from scipy.signal import detrend, butter, filtfilt, find_peaks


def jade(X, m=None, verbose=False):
    """
    Implementasi lengkap algoritma JADE untuk Independent Component Analysis (ICA).

    Parameter:
        X (np.ndarray): Matriks data input (n_channels x n_samples).
                        Diasumsikan sudah dalam kondisi centered (mean = 0).
        m (int): Jumlah komponen yang ingin diekstraksi. Jika None, maka m = n_channels.
        verbose (bool): Jika True, menampilkan informasi intermediate.

    Mengembalikan:
        np.ndarray: Estimated source signals (m x n_samples).
    """
    # Langkah 0: Centering data (jika belum)
    X = X - np.mean(X, axis=1, keepdims=True)
    n, T = X.shape
    if m is None:
        m = n

    # Langkah 1: Whitening
    R = np.dot(X, X.T) / T  # Matriks covariance
    d, E = np.linalg.eigh(R)
    # Urutkan secara descending
    idx = np.argsort(d)[::-1]
    d = d[idx]
    E = E[:, idx]
    D = np.diag(1.0 / np.sqrt(d[:m]))
    whitening_mat = D @ E[:, :m].T
    X_white = whitening_mat @ X  # Data setelah whitening (m x T)

    # Langkah 2: Estimasi cumulant matrices orde 4.
    nb = m * (m + 1) // 2  # Jumlah cumulant matrices
    CM = np.zeros((m, m, nb))
    index = []
    for i in range(m):
        for j in range(i, m):
            index.append((i, j))
    for t in range(T):
        x = X_white[:, t][:, None]  # Vector kolom (m x 1)
        xxT = x @ x.T  # Outer product (m x m)
        k = 0
        for (i, j) in index:
            CM[i, j, k] += xxT[i, j]
            if i != j:
                CM[j, i, k] = CM[i, j, k]
            k += 1
    CM /= T
    # Mengurangi Gaussian part.
    k = 0
    for i, j in index:
        if i == j:
            CM[i, j, k] -= 3
        else:
            CM[i, j, k] -= 1
            CM[j, i, k] = CM[i, j, k]
        k += 1

    # Langkah 3: Joint diagonalization menggunakan iterative Givens rotations.
    V = np.eye(m)  # Matriks rotasi yang akan dicari
    eps = 1e-6
    encore = True
    if verbose:
        print("Memulai joint diagonalization...")
    while encore:
        encore = False
        for p in range(m - 1):
            for q in range(p + 1, m):
                g11 = 0.0
                g12 = 0.0
                for k in range(nb):
                    g11 += CM[p, p, k] - CM[q, q, k]
                    g12 += CM[p, q, k] + CM[q, p, k]
                theta = 0.5 * np.arctan2(2 * g12, g11 + 1e-12)  # Menambahkan konstanta kecil untuk menghindari pembagian nol
                c = np.cos(theta)
                s = np.sin(theta)
                if np.abs(s) > eps:
                    encore = True
                    # Update matriks rotasi V (rotasi kolom p dan q)
                    temp = c * V[:, p] + s * V[:, q]
                    V[:, q] = -s * V[:, p] + c * V[:, q]
                    V[:, p] = temp
                    # Rotasi cumulant matrices untuk indeks p dan q
                    for k in range(nb):
                        temp_p = c * CM[p, :, k] + s * CM[q, :, k]
                        temp_q = -s * CM[p, :, k] + c * CM[q, :, k]
                        CM[p, :, k] = temp_p
                        CM[q, :, k] = temp_q
                        temp_p = c * CM[:, p, k] + s * CM[:, q, k]
                        temp_q = -s * CM[:, p, k] + c * CM[:, q, k]
                        CM[:, p, k] = temp_p
                        CM[:, q, k] = temp_q
        if verbose:
            print("Iterasi selesai, encore =", encore)
    if verbose:
        print("Joint diagonalization selesai.")
    # Matriks separating (demixing)
    W = V.T @ whitening_mat
    # Estimated source signals
    S = W @ X
    return S

def process_signal(V_RGB):
    """
    Memproses sinyal raw mean RGB dari frame video wajah.

    Fungsi ini mengimplementasikan:
      1. Signal amplification (mengalikan dengan 1000)
      2. Detrending (menghilangkan drift baseline yang lambat)
      3. High-pass filtering (menghilangkan tren frekuensi rendah)
      4. Normalization (zero mean dan unit variance)
      5. Ekstraksi channel green dan (opsional) blind source separation via JADE

    Parameter:
        V_RGB (np.ndarray): Sinyal raw mean RGB dengan bentuk (3, N).

    Mengembalikan:
        np.ndarray: Pulse signal yang diekstraksi dari channel green (1D array dengan panjang N).
    """
    # Langkah 1: Signal Amplification
    V_amp = V_RGB * 1000

    # Langkah 2: Detrending sepanjang waktu (axis=1)
    V_detrended = detrend(V_amp, axis=1)

    # Langkah 3: High-pass filtering.
    fs = 30.0       # Sampling frequency dalam Hz
    cutoff = 0.5    # Cutoff frequency dalam Hz
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(2, normal_cutoff, btype='high', analog=False)
    V_filtered = filtfilt(b, a, V_detrended, axis=1)

    # Langkah 4: Normalization untuk tiap channel (zero mean, unit variance)
    mean_vals = np.mean(V_filtered, axis=1, keepdims=True)
    std_vals = np.std(V_filtered, axis=1, keepdims=True)
    V_norm = (V_filtered - mean_vals) / std_vals

    # Ekstrak channel green (diasumsikan V_RGB berurutan [R; G; B])
    V_green = V_norm[1:2, :]  # Bentuk: (1, N)

    # Opsional: Terapkan JADE pada channel green.
    # Catatan: Untuk input single-channel, JADE tidak akan mengubah sinyal,
    # namun disertakan untuk mencerminkan deskripsi pada paper.
    S = jade(V_green, m=1, verbose=False)
    pulse_signal = S.flatten()
    return pulse_signal

def detect_peaks_valleys(signal, fs=20.0, min_interval_sec=0.25):
    """
    Mendeteksi peaks dan valleys dalam sinyal pulse 1D.

    Menggunakan SciPy's find_peaks untuk mendeteksi local maxima (peaks) 
    dan mendeteksi local minima (valleys) dengan membalik sinyal,
    serta menerapkan jarak minimum berdasarkan sampling frequency.

    Parameter:
        signal (np.ndarray): Sinyal pulse 1D.
        fs (float): Sampling frequency dalam Hz (default 20 Hz).
        min_interval_sec (float): Interval minimum dalam detik antara peaks dan valleys (default 0.25 detik).

    Mengembalikan:
        tuple: (peaks, valleys) dimana masing-masing merupakan array indeks.
    """
    min_interval_samples = int(min_interval_sec * fs)
    peaks, _ = find_peaks(signal, distance=min_interval_samples)
    valleys, _ = find_peaks(-signal, distance=min_interval_samples)
    return peaks, valleys

def compute_E_peak_valley(signal, peaks, valleys):
    """
    Menghitung rata-rata nilai peak (E_peak) dan rata-rata nilai valley (E_valley).

    Parameter:
        signal (np.ndarray): Sinyal pulse 1D.
        peaks (np.ndarray): Indeks-indeks peaks yang terdeteksi.
        valleys (np.ndarray): Indeks-indeks valleys yang terdeteksi.

    Mengembalikan:
        tuple: (E_peak, E_valley) sebagai nilai rata-rata dari peaks dan valleys.
    """
    E_peak = np.mean(signal[peaks]) if len(peaks) > 0 else None
    E_valley = np.mean(signal[valleys]) if len(valleys) > 0 else None
    return E_peak, E_valley


def cpu_POS(signal, **kargs):
    eps = 10**-9
    X = signal
    e, c, f = X.shape
    w = int(1.6 * kargs['fps'])   # window length

    P = np.array([[0, 1, -1], [-2, 1, 1]])  # Proyeksi matriks
    Q = np.stack([P for _ in range(e)], axis=0)

    H = np.zeros((e, f))
    for n in np.arange(w, f):
        m = n - w + 1
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2) + eps)
        M = np.expand_dims(M, axis=2)
        Cn = np.multiply(M, Cn)

        S = np.dot(Q, Cn)[0]
        S = np.swapaxes(S, 0, 1)
        
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

    return H

def estimate_bp(E_peak, E_valley, bmi):
    SBP = 23.7889 + 95.4335 * E_peak + 4.5958 * bmi - 5.109 * E_peak * bmi
    DBP = -17.3772 - 115.1747 * E_valley + 4.0251 * bmi + 5.2825 * E_valley * bmi
    return SBP, DBP

# # -----------------------
# # Contoh penggunaan:
# if __name__ == "__main__":
#     # Simulasi sinyal raw mean RGB (array 3 x N)
#     N = 600  # Misalnya, 30 detik pada 20 fps
#     t = np.linspace(0, 30, N)
#     # Buat sinyal simulasi dengan pulse sinusoidal pada tiap channel.
#     V_R = 0.5 + 0.01 * np.sin(2 * np.pi * 1.2 * t)
#     V_G = 0.5 + 0.015 * np.sin(2 * np.pi * 1.2 * t + 0.1)
#     V_B = 0.5 + 0.008 * np.sin(2 * np.pi * 1.2 * t - 0.1)
#     V_RGB = np.vstack([V_R, V_G, V_B])
    
#     # Proses sinyal (Langkah 1-4 dan ekstraksi channel green via JADE)
#     pulse_signal = process_signal(V_RGB)
    
#     # Langkah 5: Deteksi peaks dan valleys pada sinyal pulse yang diekstraksi
#     peaks, valleys = detect_peaks_valleys(pulse_signal)
    
#     # Langkah 6: Hitung nilai rata-rata peak dan valley (E_peak dan E_valley)
#     E_peak, E_valley = compute_E_peak_valley(pulse_signal, peaks, valleys)
    
#     print("Indeks peaks terdeteksi:", peaks)
#     print("Indeks valleys terdeteksi:", valleys)
#     print("E_peak (nilai rata-rata peak):", E_peak)
#     print("E_valley (nilai rata-rata valley):", E_valley)
