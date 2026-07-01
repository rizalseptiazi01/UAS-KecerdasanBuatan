# Panduan Pengerjaan Proyek Klasifikasi Gerakan Shalat 

Repositori ini berisi sistem cerdas klasifikasi pose gerakan shalat menggunakan ekstraksi koordinat MediaPipe dan jaringan saraf tiruan (ANN).

## 🛠️ Langkah Pengerjaan Proyek

1. **Ekstraksi Dataset:** Menghubungkan Google Drive ke Google Colab, mengekstrak file zip gambar mentah gerakan shalat ke direktori lokal sementara.
2. **Exploratory Data Analysis (EDA):** Melakukan perhitungan jumlah file gambar di setiap folder kelas untuk memastikan keseimbangan sebaran data.
3. **Feature Engineering (MediaPipe):** Memproses seluruh gambar lewat `cv2` dan `mediapipe.solutions.pose`. Membaca 33 titik sendi tubuh, mengambil nilai koordinat $(x, y, z, v)$, lalu menyimpannya ke dalam file `dataset_koordinat_shalat.csv`.
4. **Data Pre-processing:** * Memisahkan kolom fitur ($X$) dan kolom label ($Y$).
   * Mengubah label teks menjadi indeks angka via `LabelEncoder` dan biner kategorial via `to_categorical`.
   * Membagi data dengan proporsi **80% Data Latih (425 sampel)** dan **20% Data Uji (107 sampel)** menggunakan `train_test_split` (stratified).
5. **Model Training (TensorFlow/Keras):** Membangun arsitektur ANN Sekuensial (Input Layer 132 neuron $\rightarrow$ Hidden Layer 128 & 64 neuron dengan Dropout 0.3 & BatchNormalization $\rightarrow$ Output Layer 7 neuron Softmax). Model dilatih dengan Adam Optimizer selama 60 epoch.
6. **Evaluasi Akhir:** Menguji model dengan data uji menggunakan *Classification Report* (mencapai akurasi makro 90%) dan *Confusion Matrix* untuk melihat performa detail klasifikasi.
