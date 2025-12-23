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
Proyek ini memanfaatkan model Convolutional Neural Networks (CNN), termasuk model pretrained seperti EfficientNetB0 dan MobileNetV3 Small, untuk mengenali berbagai jenis penyakit tanaman secara otomatis.

---

Aplikasi ini bertujuan untuk:  
- Membantu petani atau peneliti mendeteksi penyakit tanaman lebih cepat dan akurat.  
- Menjadi prototipe penelitian yang bisa dikembangkan lebih lanjut untuk sistem pertanian cerdas.  
- Memberikan visualisasi probabilitas prediksi untuk setiap kelas penyakit.

### Fitur Utama
- Upload gambar daun tanaman (format JPG / PNG)  
- Prediksi penyakit dengan confidence score  
- Visualisasi probabilitas prediksi untuk semua kelas  
- Mendukung berbagai model CNN (CNN Base, EfficientNetB0, MobileNetV3 Small)  

### Dataset
Dataset yang digunakan diambil dari **[High Quality Crop Disease Image Dataset](https://www.kaggle.com/datasets/akarshangupta/high-quality-crop-disease-image-dataset-for-cnns)** yang terdiri dari 134 kelas dan 50.000 lebih gambar.