# <h1 align="center">🌿 Plant Disease Classification</h1>
<div align="center">
  <img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-023-46218-5/MediaObjects/41598_2023_46218_Fig2_HTML.jpg" alt="Gambar Utama" width="500" height="300">
  <p>
    <small>
      Sumber Image : <a href="https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-023-46218-5/MediaObjects/41598_2023_46218_Fig2_HTML.jpg">Access Here.....</a>
    </small>
  </p>
</div>

<h1 align="center"> Deskripsi Project </h1>

Plant Disease Classification adalah aplikasi berbasis *Deep Learning* yang dirancang untuk mengklasifikasikan penyakit pada tanaman dari berdasarkan citra. 
Proyek ini memanfaatkan model Convolutional Neural Networks (CNN), termasuk model pretrained seperti EfficientNetB0 dan MobileNetV3Small, untuk mengenali berbagai jenis penyakit tanaman secara otomatis.

---

Aplikasi ini bertujuan untuk:  
- Membangun model *klasifikasi penyakit pada tanaman* untuk memprediksi penyakit pada tanaman bedasarkan citra.
- *Evaluasi performa* dengan menguji beberapa model Deep Learning seperti Convolutional Neural Networks (CNN), EfficientNetB0, dan MobileNetV3Small.
- Membangun aplikasi berbasis web dengan menggunakan *Streamlit*.

## Fitur Utama
- Upload gambar daun tanaman (format JPG / PNG)  
- Prediksi penyakit dengan confidence score  
- Visualisasi probabilitas prediksi untuk semua kelas  
- Mendukung berbagai model CNN (CNN Base, EfficientNetB0, MobileNetV3Small)  

---

## Dataset
Dataset yang digunakan diambil dari **[High Quality Crop Disease Image Dataset](https://www.kaggle.com/datasets/akarshangupta/high-quality-crop-disease-image-dataset-for-cnns)** 
- 134 kelas
- ± 50.000 gambar.

Sebelum menjalankan Jupyter perhatikan struktur folder berikut :

project-root/
│
├─ data/    ← Letakkan file ZIP dataset di sini sebelum menjalankan Jupyter
├─ metadata/
├─ splits/
├─ models/
├─ app.py
├─ Crop_Disease_Classification.ipynb
└─ ...

## Persiapan Dataset
Dilakukan filtering pada dataset karena beberapa alasan berikut :
- Dataset awal terdiri dari 134 kelas, tetapi setiap kelas memiliki jumlah gambar yang berbeda-beda. Beberapa kelas mungkin hanya memiliki puluhan gambar, sementara kelas lain ratusan atau ribuan.
- Jumlah kelas yang terlalu banyak juga bisa menyebabkan proses training lambat serta memakan memori besar.
- Filtering dilakukan dengan mekanisme :
    - Filter kelas dengan jumlah gambar ≥ 100.
    - Dari kelas yang lolos filter, pilih 50 kelas dengan gambar terbanyak.
    - Salin gambar dari kelas terpilih ke folder baru data/dataset_filtered.
    - Simpan metadata kelas dan jumlah gambar untuk referensi dan reproducibility.

Hal ini juga dilakukan berdasarkan keterbatasan device yang digunakan sehingga filtering ditujukan untuk menjaga *Stabilitas Model*, *Efisiensi Komputasi*, *Representatif Data*, dan *Mengurangi Noise*.

---

## Preprocessing

Sebelum dilakukan klasifikasi, dataset melalui beberapa proses Preprocessing berikut:
- *Filtering Class* : Melakukan Filtering dengan metode mengatur *Tresshold* hanya kelas yang memiliki ≥100 gambar yang dipertahankan, dan memilih *50 kelas dengan gambar terbanyak*. lalu menyimpan metaadata yang berisi daftar kelas dan jumlah gambar untuk reproduksibilitas.

- *Splitting Dataset* : Membuat pembagian data dengan rasio 80% Training, 10% Validation, dan 10% Testing dari dataset yang sudah difilter. Output yang tercipta adalah 3 CSV yang berisi filepath dan label untuk masing-masing subset.

- *Label Encoding* : Mengubah label kategori menjadi Interger ID (label_id) agar bisa digunakan model, lalu menyimpan hasil mapping label ke dalam bentuk JSON untuk prediksi di streamlit.\

- *Preprocessing Model* : Pada tahap ini terdapat 2 jenis preprocessing yang dilakukan yakni :

    - Preprocessing CNN Base (No Pretrained) : 
        - *Resize* ke ukuran (224×224).
        - *Normalisasi* pixel ke rentang [0, 1].
        - Dataset diubah menjadi *tf.data.Dataset* untuk efisiensi.

    - Preprocessing CNN Pretrained (EfficientNetB0 dan MobileNetV3Small) : 
        - *Class weight* untuk mengatasi ketidakseimbangan kelas saat training model.
        - *Resize* ke ukuran (224×224).
        - Base dataset dibuat sebagai *tf.data.Dataset* untuk efisiensi pipeline.
        - *EfficientNetB0*: menggunakan preprocess_input dari tensorflow.keras.applications.efficientnet
        - *MobileNetV3Small*: menggunakan preprocess_input dari tensorflow.keras.applications.mobilenet_v3

---

## Model Yang Digunakan

**1. CNN Base Light**
Model ini merupakan CNN custom ringan yang dibangun dari awal (tanpa pretrained weights).  
- **Cara kerja:** Memproses citra daun melalui beberapa blok konvolusi untuk mengekstrak fitur visual, kemudian fitur-fitur tersebut diratakan dengan GlobalAveragePooling dan diteruskan ke dense layer untuk prediksi kelas penyakit.  
- **Tujuan:** Memberikan prediksi penyakit tanaman secara efisien pada dataset terfilter (50 kelas) dengan sumber daya komputasi terbatas. Ringan, cepat dilatih, dan mudah diintegrasikan dengan pipeline preprocessing yang ada. 

**Arsitektur:**
- Input: `(128, 128, 3)`
- 3 Convolutional Blocks:
  - Setiap blok: 2 × Conv2D lalu melalui layer BatchNormalization, lalu layer ReLU, dan terakhir layer MaxPooling.
  - Jumlah filter meningkat dari 32 → 64 → 128 per blok untuk menangkap fitur lebih kompleks.
- GlobalAveragePooling2D : meratakan fitur sebelum masuk ke dense layer.
- Dense Layer:
  - 256 neuron + ReLU + Dropout 0.5 → mengurangi overfitting

**Hasil Training Model**:

![Accuracy and Loss CNN-Base](output/Accuracy_Loss_CNN-Base.png)

---

**2. EfficientNetB0 (Fine-Tuned)** 
- **Cara kerja:**  
  Mengambil fitur visual dari citra daun melalui backbone EfficientNetB0, lalu menambahkan **classification head** yang terdiri dari batch normalization, dense layer, dan dropout sebelum output softmax.  
  Model hanya melakukan fine-tuning pada 15% layer terakhir, sedangkan layer awal dibekukan untuk mempertahankan fitur pretrained.  

- **Tujuan:**  
  Memanfaatkan transfer learning untuk meningkatkan akurasi klasifikasi penyakit tanaman, terutama saat dataset relatif kecil (50 kelas), tanpa perlu melatih model dari awal.  

**Arsitektur Head:**
- BatchNormalization → menstabilkan aktivasi
- Dense(256, ReLU, L2 regularization) → ekstraksi fitur lanjutan
- Dropout(0.3) → mengurangi overfitting
- Dense(NUM_CLASSES, softmax) → output probabilitas per kelas

**Hasil Training Model**
![Accuracy and Loss EfficientNetB0](output/Accuracy_Loss_EfficientNetB0.png)

### 3. MobileNetV3 Small (Pretrained)
- **Cara kerja:** Layer awal pretrained dibekukan, sedangkan head custom dilatih untuk dataset lokal. Memproses citra dengan model ringan, cepat, dan efisien.  
- **Tujuan:** Memberikan prediksi cepat untuk perangkat dengan sumber daya terbatas, tetap memanfaatkan kekuatan transfer learning.

**Arsitektur Head Custom:**
- Dense 128 neuron + ReLU + Dropout 0.4  
- Dense 50 neuron + Softmax  

**Hasil Training Model:**  
![Accuracy and Loss MobileNetV3 Small](path_to_your_output/Accuracy_Loss_MobileNetV3Small.png)
