# Laporan UAS Kecerdasan Buatan: Klasifikasi Gerakan Shalat Menggunakan MediaPipe dan Artificial Neural Network (ANN)

### Teknik Informatika A

### **Anggota Kelompok 8:**
1. Rizal Septiazi - 2406034
2. M Rafli R - 2406108

### **Domain Proyek (Latar Belakang):**
Gerakan shalat merupakan serangkaian aktivitas fisik yang memiliki urutan dan posisi tubuh yang baku. Kesalahan posisi atau ketidaksesuaian gerakan sering kali terjadi bagi mualaf atau anak-anak yang sedang belajar shalat.(Rahman, M. M., Alharazi, R. A. A., & Badri, M. K. I. b. Z. (2023). Pemanfaatan Artificial Intelligence berbasis Computer Vision dapat membantu mendeteksi kebenaran gerakan tersebut.(Suteja, J., & Setiawan, B. (2025). Namun, jika menggunakan gambar mentah secara langsung (seperti metode CNN), sistem sering kali terkecoh oleh faktor latar belakang ruangan (*background noise*) atau warna pakaian aktor, serta membutuhkan daya komputasi yang sangat berat untuk dijalankan secara *real-time*. Proyek ini hadir untuk mengatasi masalah tersebut dengan memetakan struktur tubuh menjadi titik koordinat spatial sebelum diklasifikasikan oleh kecerdasan buatan. (Miftahudin et al., 2025).

---

## 1. Business Understanding

* **Permasalahan Dunia Nyata dan Literatur Review:**
    Pengenalan visual berbasis gambar mentah (*raw images*) menggunakan Convolutional Neural Network (CNN) pada dataset terbatas rawan mengalami *overfitting* dan sensitif terhadap perubahan lingkungan (pencahayaan, background). Eksperimen awal menggunakan CNN menunjukkan **akurasi mandek di angka 41%** dan menyebabkan performa aplikasi web *lagging*. Diperlukan pendekatan ekstraksi fitur tubuh (*skeleton extraction*) untuk memisahkan objek manusia dari latar belakangnya.
* **Tujuan Proyek:**
    Membangun model klasifikasi gerakan shalat yang ringan, cepat, dan adaptif untuk diimplementasikan secara *real-time* dengan tingkat akurasi di atas 85%.
* **Siapa User/Pengguna Sistem:**
    Mualaf, anak-anak yang sedang belajar shalat, maupun instruktur agama sebagai alat bantu koreksi gerakan shalat otomatis.
* **Solusi dan Manfaat Implementasi AI:**
    Solusi yang diajukan adalah mengombinasikan **MediaPipe Pose Landmarker** (ekstraktor fitur) dengan **Artificial Neural Network (ANN)**. Manfaatnya, komputasi menjadi jauh lebih ringan untuk pelacakan kamera web secara langsung dan sistem tetap konsisten mendeteksi gerakan terlepas dari warna pakaian atau lokasi pengguna. (Igiri, C. P., Anyama, O. U., & Ita, S. A. (2015).

---

## 2. Data Understanding

* **Sumber Data:**
    Sumber dalam pengumpulan data didapat dari Kaggle.com yang dimana dengan menggunakan kata kunci "Salat Posture", dengan nama dataset ialah IMCSPD (Islamic Multi-Category Salat Posture Dataset). Di dalamnya terdapat banyak gambar-gambar mengenai berbagai posture gerakan shalat. Total jumlah sampel/label pada dataaset ini sekitar 25 sampel/label, dan yang difokuskan hanya 7 sampel/label.(Sumber Kaggle IMCSPD:https://www.kaggle.com/datasets/baizidkamruzzaman19/imcspd).
* **Ukuran dan Format Data:**
    * **Data Awal:** Berupa ratusan file gambar yang tersebar ke dalam 7 folder kelas gerakan. Total sampel yang siap diekstrak adalah 532 gambar (425 data latih dan 107 data uji).
    * **Data Akhir:** Berupa berkas tabel berekstensi `.csv` bernama `dataset_koordinat_shalat.csv`.
* **Tipe Data dan Target Klasifikasi:**
    Tipe data input adalah data numerik (berupa koordinat *floating-point*). Target klasifikasi terdiri dari **7 kelas gerakan shalat**, yaitu:
  
    1. Jalsa
    2. Qiyam_Recitation
    3. Ruku
    4. Salam_Left
    5. Salam_Right
    6. Sujud
    7. Takbir
* **Deskripsi Setiap Fitur (Atribut):**
    Setiap baris data mewakili 1 gambar yang dipecah oleh MediaPipe menjadi **33 titik sendi utama tubuh** (bahu, siku, pergelangan tangan, lutut, tumit, dll). Setiap 1 titik sendi memiliki 4 parameter atribut:
    * `x`: Posisi horizontal sendi (kanan-kiri) dalam ruang kamera.
    * `y`: Posisi vertikal sendi (atas-bawah) dalam ruang kamera.
    * `z`: Kedalaman posisi sendi (maju-mundur/dimensi 3D).
    * `v`: *Visibility* (Tingkat kejelasan visual sendi dari halangan objek lain).
    Total fitur input dalam CSV adalah **132 kolom fitur** ($33 \text{ sendi} \times 4 \text{ atribut}$).

---

## 3. Visualisasi Distribusi Data (EDA)

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
* **Analisis Korelasi Antar Fitur (Heatmap / Pairplot):**
  Pada proyek ini, analisis korelasi linier tradisional (seperti Pearson Correlation Heatmap atau Pairplot) **tidak diimplementasikan**. Hal ini didasarkan pada karakteristik unik dari data koordinat spasial MediaPipe:
  1. **Jumlah Fitur Terlalu Besar (High Dimensionality):** Dataset memiliki 132 fitur kolom angka numerik. Membuat *Pairplot* atau *Heatmap* berukuran $132 \times 132$ akan menghasilkan visualisasi yang sangat padat, tidak terbaca, dan tidak memberikan informasi yang bermakna (*uninterpretable*).
  2. **Korelasi Bersifat Non-Linier Dinamis:** Hubungan antar-sendi tubuh (misalnya korelasi antara posisi koordinat tangan $x\_15$ dan koordinat lutut $y\_25$) berubah secara drastis tergantung pada jenis gerakan shalatnya (saat berdiri vs saat sujud). Korelasi statis linier tidak mampu menggambarkan perubahan geometris tubuh ini secara global.
  3. **Pendekatan Ekstraksi Ciri Langsung oleh Jaringan Saraf:** Tugas untuk mencari korelasi spasial, jarak antar-sendi, dan sudut lekukan tubuh diserahkan sepenuhnya secara otomatis kepada lapisan tersembunyi (*Hidden Layers*) pada arsitektur ANN melalui perhitungan matriks bobot (*weights*), sehingga analisis korelasi manual di awal tidak lagi diperlukan.
* **Deteksi Data Tidak Seimbang (Imbalanced Classes):**
  Dari visualisasi terlihat adanya sedikit perbedaan jumlah sampel antar kelas, di mana kelas `Salam_Left` memiliki jumlah data paling sedikit (di bawah 50 sampel) dibandingkan kelas `Jalsa` yang paling dominan. Namun, tingkat ketimpangan ini masih dalam batas aman (tidak ekstrem) sehingga tidak memerlukan teknik *Oversampling* (seperti SMOTE).

* **Insight Awal dari Pola Data:**
  Meskipun jumlah gambar bervariasi, pola spasial dari 33 titik koordinat sendi yang diekstrak oleh MediaPipe bersifat unik untuk setiap pose. Hal ini memungkinkan algoritma pengenal pola mengenali ciri khas sudut tubuh secara konsisten pada tahap pemodelan.

---

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

---

## 5. Modeling

* **Pemilihan Algoritma:**
  Proyek ini memilih dan mengimplementasikan **Artificial Neural Network (ANN)** Sekuensial setelah mengevaluasi keterbatasan **Convolutional Neural Network (CNN)**.
* **Alasan Pemilihan Algoritma:**
  1. **CNN (Dibatalkan):** Eksperimen awal menggunakan CNN langsung pada gambar mentah menghasilkan **akurasi yang mandek di angka 41%**. CNN sangat rentan terhadap *overfitting* karena keterbatasan jumlah data gambar, memakan daya komputasi yang berat (lagging di Streamlit), serta sensitif terhadap *background noise*.(Mohammadpour, L, et al., 2022)
  2. **ANN (Dipilih):** Setelah data diekstrak menjadi koordinat 132 fitur oleh MediaPipe, input data berubah menjadi numerik bersih. ANN sangat optimal, efisien, dan ringan untuk memproses data angka koordinat tersebut, menjadikannya sangat responsif untuk kebutuhan deteksi *real-time*. (Tzeico J. Sánchez-Vicinaiz et al,.2024).
* **Implementasi Model (Arsitektur Jaringan):**
  Model ANN dibangun menggunakan TensorFlow/Keras dengan arsitektur sekuensial:
  * *Input Layer*: 132 neuron (menerima 33 sendi $\times$ 4 atribut).
  * *Hidden Layer 1*: Dense 128 neuron + BatchNormalization + Dropout (0.3).
  * *Hidden Layer 2*: Dense 64 neuron + Dropout (0.3).
  * *Output Layer*: Dense 7 neuron dengan fungsi aktivasi *Softmax*.
  Model dikompilasi menggunakan *Adam Optimizer* dan *Categorical Crossentropy*, lalu dilatih selama 60 epoch.
* **Perbandingan Model:**
  Dilakukan eksperimen perbandingan antara dua pendekatan model Deep Learning, yaitu *Convolutional Neural Network* (CNN) berbasis gambar mentah dan *Artificial Neural Network* (ANN) berbasis koordinat spasial:

  | Parameter Pembanding | Pendekatan 1: CNN (Eksperimen Awal) | Pendekatan 2: MediaPipe + ANN (Model Terpilih) |
  | :--- | :--- | :--- |
  | **Input Data** | Gambar mentah / matriks piksel visual ($224 \times 224$) | 132 Fitur Angka (Koordinat 3D 33 Titik Sendi) |
  | **Akurasi Akhir** | **41%** (Mandek / Macet) | **90%** (Sangat Tinggi dan Stabil) |
  | **Kondisi Model** | *Overfitting* (Hanya menghafal dataset latihan) | *Konvergen* (Sehat dan mampu digeneralisasikan) |
  | **Beban Komputasi** | Sangat Berat (Memproses jutaan piksel tiap detik) | Sangat Ringan (Hanya memproses baris angka numerik) |
  | **Sensitivitas Lingkungan** | Tinggi (Terkecoh oleh warna baju & *background*) | Kebal (Hanya fokus pada ekstraksi struktur sendi tubuh) |
  | **Performa Real-Time** | *Lagging* / Patah-patah pada web Streamlit | Responsif, lancar, dan *Anti-Lag* |

* **Analisis Hasil Perbandingan:**
  Berdasarkan tabel eksperimen di atas, model **MediaPipe + ANN** unggul telak di segala parameter dibandingkan CNN biasa. CNN mengalami kegagalan (akurasi hanya 41%) karena keterbatasan jumlah dataset gambar mentah yang memaksa model mengalami *overfitting*, ditambah gangguan *background noise* ruangan. 
  
  Sebaliknya, dengan memindahkan tugas ekstraksi visual ke MediaPipe dan menggunakan ANN sebagai otak klasifikasi angka koordinatnya, akurasi berhasil melonjak drastis hingga **90%**. Komputasi yang ringkas membuat model ANN menjadi opsi paling rasional dan terbaik untuk diimplementasikan ke dalam sistem deteksi kamera secara *real-time*.

* **Visualisasi Model:**
  Visualisasi arsitektur model *Artificial Neural Network* (ANN) yang dibangun menampilkan struktur berlapis (*layered architecture*) yang menggambarkan bagaimana data fitur koordinat sendi tubuh mengalir dan diproses hingga menghasilkan keputusan kelas gerakan shalat:

  ```text
  [Input Layer: 132 Fitur Koordinat]
                 │
                 ▼
  [Dense Layer: 128 Neuron + ReLU]  ──► (Ekstraksi pola spatial sendi)
                 │
                 ▼
  [Batch Normalization Layer]       ──► (Stabilisasi skala nilai aktivasi)
                 │
                 ▼
  [Dropout Layer: 0.3]              ──► (Regulasi acak untuk anti-overfitting)
                 │
                 ▼
  [Dense Layer: 64 Neuron + ReLU]   ──► (Penyederhanaan fitur tak berwujud)
                 │
                 ▼
  [Dropout Layer: 0.3]              ──► (Regulasi acak tahap kedua)
                 │
                 ▼
  [Output Layer: 7 Neuron + Softmax]──► (Probabilitas 7 Kelas Gerakan Shalat)
**Penjelasan Aliran Visualisasi:**
* Input Layer: Menerima 132 nilai angka numerik representasi 33 titik sendi (X, Y, Z, V) dari MediaPipe.
* Hidden Layer (Dense 128 & 64): Lapisan saraf tersembunyi berstruktur Fully Connected yang bertugas mengalkulasi bobot (weights) dan bias untuk mengenali kecenderungan sudut tubuh unik di setiap pose shalat.
* Regularization Layer (Batch Normalization & Dropout): Berfungsi menjaga stabilitas komputasi dan memotong sebagian koneksi saraf secara acak sebesar 30% saat latihan agar model tidak bias atau mengalami overfitting.
* Output Layer: Lapisan akhir yang memetakan hasil komputasi menjadi 7 output kelas menggunakan fungsi Softmax untuk mengeluarkan nilai probabilitas tertinggi gerakan shalat (Jalsa, Ruku, Sujud, dll.).
---

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

## 8. Kesimpulan dan Rekomendasi

* **Ringkasan Hasil Modeling dan Evaluasi:**
  Penelitian ini berhasil mengintegrasikan framework *MediaPipe Pose Landmarker* sebagai ekstraktor fitur spasial dengan algoritma *Artificial Neural Network* (ANN) sekuensial untuk mengklasifikasikan gerakan shalat. Melalui proses training selama 60 epoch, model mampu memetakan 132 fitur koordinat 3D dari 33 titik sendi tubuh manusia dengan sangat efisien. Hasil evaluasi akhir menggunakan data uji menunjukkan performa yang sangat memuaskan, di mana model berhasil mencapai nilai **Akurasi Makro (Macro Accuracy) sebesar 90%** serta menunjukkan grafik training yang konvergen dan stabil (nilai loss di bawah 0.5).

* **Apakah Tujuan Proyek Tercapai?**
  **Ya, tujuan proyek telah tercapai sepenuhnya.** Proyek ini berhasil menjawab permasalahan mendasar dari eksperimen awal menggunakan metode CNN konvensional yang sempat mengalami kegagalan akibat *overfitting* dan keterbatasan daya komputasi (akurasi mandek di 41%). Dengan beralih ke arsitektur ANN berbasis data koordinat numerik bersih, sistem terbukti tidak hanya melonjak drastis akurasinya menjadi 90%, namun juga berhasil memangkas beban komputasi secara signifikan sehingga aplikasi dapat berjalan lancar tanpa interupsi (*anti-lag*) pada perangkat dengan spesifikasi standar.

* **Kelebihan dan Keterbatasan Model:**
  * **Kelebihan:**
    1. *Kebal terhadap Efek Lingkungan (Robustness):* Karena input data ANN berupa angka koordinat skeleton tubuh hasil ekstraksi MediaPipe, model tidak akan terkecoh oleh variasi latar belakang ruangan (*background noise*), kondisi pencahayaan, maupun warna dan corak pakaian yang dikenakan oleh pengguna.
    2. *Komputasi Sangat Ringan:* Model tidak perlu melakukan kalkulasi matriks pixel gambar mentah yang berat tiap detiknya, sehingga proses inferensi (*real-time tracking* via webcam) berjalan sangat responsif dan efisien.
  * **Keterbatasan:**
    1. *Ketergantungan terhadap Keterlihatan Tubuh (Full-Body View):* Model sangat sensitif terhadap keutuhan visual objek manusia. Jika kamera terlalu dekat atau terpotong (misal bagian kaki atau pinggang terhalang objek lain), MediaPipe akan gagal menangkap koordinat sendi secara lengkap, yang berpotensi menurunkan akurasi prediksi ANN secara drastis.
    2. *Prediksi Bersifat Statis Per Frame:* Model saat ini mendeteksi pose shalat berbasis *frame-by-frame* secara instan, sehingga belum mampu membaca kontinuitas transisi gerakan dinamis atau menghitung jumlah rakaat shalat secara runtut berdasarkan durasi waktu.

* **Rekomendasi Perbaikan untuk Pengembangan Selanjutnya:**
  1. *Ekspansi dan Variasi Dataset:* Menambahkan jumlah sampel gambar baru, khususnya pada kelas yang memiliki distribusi data minimal seperti `Salam_Left`, serta memperkaya variasi sudut pengambilan kamera (sudut pandang serong/diagonal dan samping).
  2. *Implementasi Algoritma Berbasis Waktu (Time-Series):* Untuk pengembangan sistem di masa depan, disarankan untuk mengombinasikan MediaPipe dengan algoritma **LSTM (Long Short-Term Memory)** atau **GRU**. Hal ini penting agar AI dapat membaca runtunan video gerakan shalat secara kontinu berbasis waktu, sehingga sistem mampu mendeteksi kesalahan transisi gerakan serta melakukan perhitungan (*counting*) jumlah rakaat shalat secara otomatis dan cerdas.

## 9. Referensi
* Miftahuddin, F., Musthofa, A., Pratama, A. A., Syifasultana, D. and Al Mumtaz, F. J., 2025. Identification of prayer movements using Convolutional Neural Network classification model and Prewitt and morphology image processing. MALCOM: Indonesian Journal of Machine Learning and Computer Science, 5(1), pp. 473–483.
* Igiri, C. P., Anyama, O. U., & Ita, S. A. (2015). Effect of learning rate on Artificial Neural Network in machine learning. International Journal of Engineering Research & Technology (IJERT), 4(02), 359–363. https://www.ijert.org
* Rahman, M. M., Alharazi, R. A. A., & Badri, M. K. I. b. Z. (2023). Intelligent system for Islamic prayer (salat) posture monitoring. IAES International Journal of Artificial Intelligence (IJ-AI), 12(1), 220–231. https://doi.org/10.11591/ijai.v12.i1.pp220-231
* Suteja, J., & Setiawan, B. (2025). Strategi deep learning dalam mengembangkan kecerdasan artifisial pada pembelajaran Pendidikan Agama Islam di sekolah. Serumpun International Conference Proceedings (SICP), 1(1), 16–26.
* Mohammadpour, L., Ling, T. C., Liew, C. S., & Aryanfar, A. (2022). A survey of CNN-based network intrusion detection. Applied Sciences, 12(16), 8162. https://doi.org/10.3390/app12168162
* Tzeico J. Sánchez-Vicinaiz et al,.(2024). MediaPipe Frame and Convolutional Neural Networks-Based Fingerspelling Detection in Mexican Sign Language
## 10. Lampiran (Menampilkan Projek Deteksi dalam Web) 
* **Tampilan web**
  <img width="1274" height="640" alt="Screenshot 2026-07-11 114323" src="https://github.com/user-attachments/assets/3c53e1c4-c00e-4791-9216-91fc8e869a16" />
* **Pendeteksian**
  <img width="1205" height="522" alt="image" src="https://github.com/user-attachments/assets/81effe6c-503c-44b6-8e18-1c1e9297e267" />


