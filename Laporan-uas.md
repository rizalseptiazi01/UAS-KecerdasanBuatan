# Laporan UAS Kecerdasan Buatan: Klasifikasi Gerakan Shalat Menggunakan MediaPipe dan Artificial Neural Network (ANN)

### **Anggota Kelompok 8:**
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
    Sumber dalam pengumpulan data didapat dari Kaggle.com yang dimana dengan menggunakan kata kunci "Salat Posture", dengan nama dataset ialah IMCSPD (Islamic Multi-Category Salat Posture Dataset). Di dalamnya terdapat banyak gambar-gambar mengenai berbagai posture gerakan shalat. Total jumlah sampel/label pada dataaset ini sekitar 25 sampel/label, dan yang difokuskan hanya 7 sampel/label.(Sumber Kaggle IMCSPD:https://www.kaggle.com/datasets/baizidkamruzzaman19/imcspd).
* **Ukuran dan Format Data:**
    * **Data Awal:** Berupa ratusan file gambar yang tersebar ke dalam 7 folder kelas gerakan. Total sampel yang siap diekstrak adalah 532 gambar (425 data latih dan 107 data uji).
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

### 3. **Visualisasi Distribusi Data (EDA):**
Berikut adalah grafik sebaran jumlah gambar pada tiap kelas sebelum dilakukan ekstraksi fitur koordinat:
<img width="785" height="388" alt="Screenshot 2026-06-27 064327" src="https://github.com/user-attachments/assets/26e3e206-12bf-4b7d-97dd-da5ffb8ad025" />
Berdasarkan diagram batang distribusi jumlah gambar per kelas, sebaran data awal untuk tiap gerakan shalat adalah sebagai berikut:
  * `Jalsa`: > 110 gambar
  * `Qiyam_Recitation`: ~77 gambar
  * `Ruku`: ~86 gambar
  * `Salam_Left`: ~48 gambar
  * `Salam_Right`: ~93 gambar
  * `Sujud`: ~99 gambar
  * `Takbir`: ~75 gambar
### **Deteksi Data Tidak Seimbang (Imbalanced Classes):**
  Dari visualisasi terlihat adanya sedikit perbedaan jumlah sampel antar kelas, di mana kelas `Salam_Left` memiliki jumlah data paling sedikit (di bawah 50 sampel) dibandingkan kelas `Jalsa` yang paling dominan. Namun, tingkat ketimpangan ini masih dalam batas aman (tidak ekstrem) sehingga tidak memerlukan teknik *Oversampling* (seperti SMOTE).

### **Insight Awal dari Pola Data:**
  Meskipun jumlah gambar bervariasi, pola spasial dari 33 titik koordinat sendi yang diekstrak oleh MediaPipe bersifat unik untuk setiap pose. Hal ini memungkinkan algoritma pengenal pola mengenali ciri khas sudut tubuh secara konsisten pada tahap pemodelan.

## 4. Data Preparation

* **Pembersihan Data:**
  Pembersihan dilakukan saat proses ekstraksi titik koordinat. Gambar-gambar yang gagal dideteksi kerangka skeletonnya oleh MediaPipe secara otomatis dilewati (*dropped*) agar tidak menghasilkan nilai kosong (*null value*) di file CSV.
* **Encoding Data Kategorik:**
  * **Label Encoding:** Mengubah label teks string nama gerakan shalat menjadi indeks angka diskrit dari rentang 0-6 (`label_encoder.fit_transform`).
  * **One-Hot Encoding:** Mengonversi indeks angka tersebut menjadi matriks biner kategorial (`to_categorical`) dengan 7 kelas untuk memenuhi kebutuhan fungsi kehilangan jaringan saraf.
* **Normalisasi Data Numerik:**
  MediaPipe secara *default* telah menormalisasi seluruh koordinat $x$ dan $y$ ke dalam rentang nilai 0 hingga 1 berdasarkan dimensi piksel gambar, sehingga data numerik siap diproses tanpa memerlukan penskalaan manual tambahan.
* **Split Data:**
  Data dibagi secara proporsional dengan rasio **80% untuk Data Latih (425 sampel)** dan **20% untuk Data Uji (107 sampel)** menggunakan fungsi `train_test_split` dengan parameter `stratify=Y_encoded` guna menjamin distribusi representasi kelas yang seimbang pada kedua bagian data.

## 5. Modeling

* **Pemilihan Algoritma:**
  Proyek ini memilih dan mengimplementasikan **Artificial Neural Network (ANN)** Sekuensial setelah mengevaluasi keterbatasan **Convolutional Neural Network (CNN)**.
* **Alasan Pemilihan Algoritma:**
  1. **CNN (Dibatalkan):** Eksperimen awal menggunakan CNN langsung pada gambar mentah menghasilkan **akurasi yang mandek di angka 41%**. CNN sangat rentan terhadap *overfitting* karena keterbatasan jumlah data gambar, memakan daya komputasi yang berat (lagging di Streamlit), serta sensitif terhadap *background noise*.
  2. **ANN (Dipilih):** Setelah data diekstrak menjadi koordinat 132 fitur oleh MediaPipe, input data berubah menjadi numerik bersih. ANN sangat optimal, efisien, dan ringan untuk memproses data angka koordinat tersebut, menjadikannya sangat responsif untuk kebutuhan deteksi *real-time*.
* **Implementasi Model (Arsitektur Jaringan):**
  Model ANN dibangun menggunakan TensorFlow/Keras dengan arsitektur sekuensial:
  * *Input Layer*: 132 neuron (menerima 33 sendi $\times$ 4 atribut).
  * *Hidden Layer 1*: Dense 128 neuron + BatchNormalization + Dropout (0.3).
  * *Hidden Layer 2*: Dense 64 neuron + Dropout (0.3).
  * *Output Layer*: Dense 7 neuron dengan fungsi aktivasi *Softmax*.
  Model dikompilasi menggunakan *Adam Optimizer* dan *Categorical Crossentropy*, lalu dilatih selama 60 epoch.

## 6. Evaluation

* **Metrik Evaluasi (Classification Report):**
  
  <img width="340" height="201" alt="Screenshot 2026-07-02 115441" src="https://github.com/user-attachments/assets/63ddb8ea-3880-4adc-96a9-37fa925d3f35" />

  Berdasarkan pengujian final menggunakan 107 data uji yang belum pernah dilihat model, model ANN ini berhasil mendapatkan **Akurasi Makro (Macro Accuracy) mencapai 90% (0.90)**. Nilai *Precision, Recall,* dan *F1-Score* secara rata-rata merata tinggi di atas 85-90% untuk seluruh kelas gerakan shalat.
  
* **Confusion Matrix:**
  
  <img width="418" height="308" alt="Screenshot 2026-06-27 070154" src="https://github.com/user-attachments/assets/908e7764-8e98-47d6-8f5c-598331985b34" />

  Dari matriks kebingungan, garis diagonal utama menunjukkan tumpukan angka tebakan yang benar secara mutlak (contoh: Jalsa benar 21 kali, Qiyam 15 kali, Ruku 14 kali). 
* **Penjelasan Kinerja Model:**
  Berdasarkan grafik evaluasi di bawah, metrik performa model dapat dijabarkan sebagai berikut:

  <img width="928" height="326" alt="Screenshot 2026-06-27 065957" src="https://github.com/user-attachments/assets/6d58991b-a4e8-4190-81e6-bc838a0c8f57" />

* **Grafik Akurasi Model (Kiri):** 
  Grafik ini mengukur tingkat ketepatan model dalam mengklasifikasikan gerakan shalat. Terlihat bahwa garis hijau (*Training Accuracy*) dan garis putus-putus biru (*Validation Accuracy*) bergerak menanjak naik secara progresif dan konsisten sejak epoch awal. Pada akhir epoch ke-60, akurasi model stabil berada di rentang **85% - 90%**, yang menandakan model sukses mengenali pola gerakan dengan sangat baik.
  
* **Grafik Loss Model (Kanan):** 
  Grafik ini mengukur tingkat kekeliruan atau error prediksi yang dihasilkan oleh model. Garis merah (*Training Loss*) dan garis putus-putus oranye (*Validation Loss*) menunjukkan tren meluncur turun secara drastis dari nilai error awal di atas 1.75 hingga berhasil ditekan hingga stabil di bawah rentang **0.3 - 0.5** pada akhir epoch. Hal ini membuktikan bahwa tingkat kesalahan prediksi model sudah sangat minim.

* **Analisis Fluktuasi Garis Validasi:** 
  Terdapat dinamika fluktuatif (naik-turun) yang tampak pada garis validasi (warna biru dan oranye) di beberapa epoch pertengahan. Hal ini merupakan kondisi yang wajar dan normal terjadi karena ukuran dataset yang efisien serta adanya penerapan regularisasi **Dropout (0.3)** di dalam arsitektur model. *Dropout* sengaja menonaktifkan sebagian saraf secara acak selama training untuk mencegah model dari ketergantungan penuh (*overfitting* atau menghafal mati data latihan). Tren kedua grafik yang tetap konvergen (bertemu di titik yang sama) membuktikan bahwa model ANN ini aman, sehat, dan memiliki kemampuan generalisasi yang baik untuk pengujian real-time.
   Sedikit kesalahan klasifikasi hanya terjadi pada gerakan *Salam* yang terkadang tertebak sebagai *Jalsa* (duduk), hal ini sangat logis karena posisi anatomi sendi tubuh pada kedua gerakan tersebut memang sama-sama dilakukan dalam posisi duduk di atas lantai.
