# [Komparasi ROI Pada Citra Wajah Orang Indonesia Untuk Pengukuran Tekanan Darah Non-Kontak Berdasarkan Metode Zhou]

## Abstrak
<p align="justify"> Tekanan darah merupakan suatu nilai yang diberikan darah kepada
dinding arteri dengan satuan tekanan darah adalah milimeter
air raksa (mmHg). Ada dua jenis tekanan darah yang terjadi
pada tubuh manusia, yaitu tekanan darah sistolik dan diastolik.
Pemeriksaan tekanan darah sangat penting, akan tetapi banyak
pasien merasa pengukuran tekanan darah dengan menggunakan
manset bertegangan tidak nyaman. Untuk itu dikembangkan
pengukuran tekanan darah secara non-kontak dengan mendeteksi
sinyal rPPG pada citra wajah pasien. Metode yang digunakan
untuk mendeteksi sinyal rPPG pada pasien adalah metode
Independent Component Analysis dengan pemilihan ROI seperti
wajah, dahi, pipi kanan, pipi kiri, hidung, dan dagu. Penelitian
ini mengevaluasi kinerja berbagai ROI untuk menemukan ROI
terbaik dalam estimasi tekanan darah sistolik dan diastolik. Untuk
mengevaluasi kinerja pada tiap ROI, digunakan metrik perthitungan
MAE dan RMSE.Dengan menggunakan dataset Physio ITERA
untuk mendeteksi sinyal rPPG menggunakan algoritma ICA dan
dihtung menggunakan perhitungan Zhou, maka ROI terbaik untuk
pengukuran tekanan darah sistolik adalah dagu dengan nilai MAE
sebesar 14.7 mmHg dan nilai RMSE sebesar 17.688 mmHg.
Sedangkan untuk pengukuran tekanan darah diastolik adalah ROI
Pipi Kiri dengan nilai MAE 10.38 mmHg dan nilai RMSE sebesar
12.92 mmHg. </p>

---

## Deskripsi Singkat
<p align="justify"> Program ini Proyek ini bertujuan untuk mengestimasi tekanan darah sistolik (SBP) dan diastolik (DBP) secara non-kontak menggunakan sinyal photoplethysmography (rPPG) yang diekstrak dari video wajah. 
  Metode ini memanfaatkan analisis sinyal pada channel warna hijau dan model regresi yang melibatkan BMI untuk meningkatkan akurasi prediksi. </p>
  
---

## Fitur & Tujuan Utama
<p align="justify">
* Ekstraksi Sinyal rPPG: Mengekstrak sinyal denyut nadi dari video wajah subjek.
* Pemrosesan Sinyal: Membersihkan sinyal mentah menggunakan teknik seperti detrending, filtering, dan normalisasi.
* *stimasi Tekanan Darah: Mengimplementasikan model regresi berdasarkan fitur sinyal dan data fisiologis (BMI) untuk memprediksi nilai SBP dan DBP.
* Analisis ROI: Menganalisis dan membandingkan kualitas sinyal dari berbagai area di wajah (dahi, pipi, dll.) dengan menggunakan metrik perhitungan MAE dan RMSE
</p>

## Struktur Repositori
<p align="justify">
