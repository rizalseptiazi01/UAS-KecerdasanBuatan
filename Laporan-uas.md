# Laporan UAS Kecerdasan Buatan: Klasifikasi Gerakan Shalat Menggunakan MediaPipe dan Artificial Neural Network (ANN)

### **Nama Kelompok:**
1. Rizal Septiazi - 2406034
2. M Rafli R - 2406108

### **Domain Proyek (Latar Belakang):**
Gerakan shalat merupakan serangkaian aktivitas fisik yang memiliki urutan dan posisi tubuh yang baku. Kesalahan posisi atau ketidaksesuaian gerakan sering kali terjadi bagi mualaf atau anak-anak yang sedang belajar shalat. Pemanfaatan Artificial Intelligence berbasis Computer Vision dapat membantu mendeteksi kebenaran gerakan tersebut. Namun, jika menggunakan gambar mentah secara langsung (seperti metode CNN), sistem sering kali terkecoh oleh faktor latar belakang ruangan (*background noise*) atau warna pakaian aktor, serta membutuhkan daya komputasi yang sangat berat untuk dijalankan secara *real-time*. Proyek ini hadir untuk mengatasi masalah tersebut dengan memetakan struktur tubuh menjadi titik koordinat spatial sebelum diklasifikasikan oleh kecerdasan buatan.

---

## 1. Business Understanding

* **Permasalahan Dunia Nyata dan Literatur Review:**
    Pengenalan visual berbasis gambar mentah (*raw images*) menggunakan Convolutional Neural Network (CNN) pada dataset terbatas rawan mengalami *overfitting* dan sensitif terhadap perubahan lingkungan (pencahayaan, background). Eksperimen awal menggunakan CNN menunjukkan **akurasi mandek di angka 41%** dan menyebabkan performa aplikasi web *lagging*. Diperlukan pendekatan ekstraksi fitur tubuh (*skeleton extraction*) untuk memisahkan objek manusia dari latar belakangnya.
* **Tujuan Proyek:**
    Membangun model klasifikasi gerakan shalat yang ringan, cepat, dan adaptif untuk diimplementasikan secara *real-time* dengan tingkat akurasi di atas 85%.
* **Siapa User/Pengguna Sistem:**
    Mualaf, anak-anak yang sedang belajar shalat, maupun instruktur agama sebagai alat bantu koreksi gerakan shalat otomatis.
* **Solusi dan Manfaat Implementasi AI:**
    Solusi yang diajukan adalah mengombinasikan **MediaPipe Pose Landmarker** (ekstraktor fitur) dengan **Artificial Neural Network (ANN)**. Manfaatnya, komputasi menjadi jauh lebih ringan untuk pelacakan kamera web secara langsung dan sistem tetap konsisten mendeteksi gerakan terlepas dari warna pakaian atau lokasi pengguna.

---

## 2. Data Understanding

* **Sumber Data:**
    Data dikumpulkan melalui simulasi manual perekaman gerakan shalat dari beberapa aktor dengan variasi sudut pandang kamera dan pencahayaan.
* **Ukuran dan Format Data:**
    * **Data Awal:** Berupa ratusan file gambar (.png/.jpeg) yang tersebar ke dalam 7 folder kelas gerakan. Total sampel yang siap diekstrak adalah 532 gambar (425 data latih dan 107 data uji).
    * **Data Akhir:** Berupa berkas tabel berekstensi `.csv` bernama `dataset_koordinat_shalat.csv`.
* **Tipe Data dan Target Klasifikasi:**
    Tipe data input adalah data numerik (berupa koordinat *floating-point*). Target klasifikasi terdiri dari **7 kelas gerakan shalat**, yaitu:
    0. Jalsa
    1. Qiyam_Recitation
    2. Ruku
    3. Salam_Left
    4. Salam_Right
    5. Sujud
    6. Takbir
* **Deskripsi Setiap Fitur (Atribut):**
    Setiap baris data mewakili 1 gambar yang dipecah oleh MediaPipe menjadi **33 titik sendi utama tubuh** (bahu, siku, pergelangan tangan, lutut, tumit, dll). Setiap 1 titik sendi memiliki 4 parameter atribut:
    * `x`: Posisi horizontal sendi (kanan-kiri) dalam ruang kamera.
    * `y`: Posisi vertikal sendi (atas-bawah) dalam ruang kamera.
    * `z`: Kedalaman posisi sendi (maju-mundur/dimensi 3D).
    * `v`: *Visibility* (Tingkat kejelasan visual sendi dari halangan objek lain).
    Total fitur input dalam CSV adalah **132 kolom fitur** ($33 \text{ sendi} \times 4 \text{ atribut}$).

### **Visualisasi Distribusi Data (EDA):**
Berikut adalah grafik sebaran jumlah gambar pada tiap kelas sebelum dilakukan ekstraksi fitur koordinat:
*(Masukkan gambar diagram batang hijau kamu di sini setelah diupload ke GitHub)*
