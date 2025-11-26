# Klasifikasi Sampah: Recycle vs Non-Recycle

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange)](https://www.tensorflow.org/) [![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/) [![Colab](https://img.shields.io/badge/Google%20Colab-yellow)](https://colab.research.google.com/)

Proyek ini merupakan implementasi model pembelajaran mesin untuk klasifikasi citra sampah secara biner: **Recycle** (dapat didaur ulang) vs **Non-Recycle** (tidak dapat didaur ulang). Menggunakan **Convolutional Neural Network (CNN)** berbasis TensorFlow/Keras sebagai model utama, dibandingkan dengan model machine learning tradisional seperti Random Forest. Proyek ini dirancang untuk dijalankan di Google Colab dengan dukungan GPU (T4).

## Deskripsi
- **Tujuan**: Mengklasifikasikan gambar sampah untuk mendukung sistem pengelolaan limbah otomatis, seperti di fasilitas daur ulang.
- **Pendekatan**:
  - **Preprocessing**: EDA (Exploratory Data Analysis), augmentasi data menggunakan `ImageDataGenerator`.
  - **Model Utama**: CNN custom dengan arsitektur sederhana (Conv2D, MaxPooling, Dense layers).
  - **Model Perbandingan**: Random Forest, SVM, dll., menggunakan fitur ekstraksi dari citra (HOG atau flattened pixels).
  - **Evaluasi**: Akurasi, classification report, confusion matrix, dan perbandingan overfitting.
- **Dataset**: 399 citra (285 recycle, 114 non-recycle) dari folder Google Drive. Distribusi tidak seimbang, sehingga augmentasi diterapkan untuk mengatasi imbalance.
- **Hasil Utama**: 
  - Akurasi Test CNN: **83.54%**.
  - Overfitting: -8.86% (model generalisasi baik ke data validasi).
  - CNN unggul dibandingkan model tradisional (misalnya, RF ~75-80%).

Proyek ini cocok untuk penelitian jurnal SINTA atau prototipe aplikasi lingkungan.

## Persyaratan (Requirements)
- Python 3.10+
- TensorFlow 2.19.0
- Scikit-learn
- Matplotlib, Seaborn, NumPy, Pandas
- OpenCV (untuk pemrosesan citra opsional)

File `requirements.txt` (buat manual jika diperlukan):
```
tensorflow==2.19.0
scikit-learn
matplotlib
seaborn
numpy
pandas
joblib
```

## Instalasi dan Setup
### Di Google Colab (Direkomendasikan)
1. Buka [Google Colab](https://colab.research.google.com).
2. Upload atau mount Google Drive Anda:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Pastikan dataset ada di `/content/drive/MyDrive/Jurnal Sinta/dataset manual/` dengan subfolder:
   - `recycle/` (gambar sampah daur ulang).
   - `non-recycle/` (gambar sampah non-daur ulang).
4. Pilih Runtime > Change runtime type > GPU (T4) untuk training cepat.
5. Jalankan sel notebook secara berurutan.

### Di Lokal (Jupyter Notebook)
1. Clone repositori:
   ```bash
   git clone <repo-url>
   cd Klasifikasi_Sampah
   ```
2. Buat environment virtual:
   ```bash
   python -m venv env
   source env/bin/activate  # Linux/Mac
   # atau env\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Update path dataset di notebook: Ganti `/content/drive/MyDrive/Jurnal Sinta/dataset manual` dengan path lokal Anda.
5. Jalankan: `jupyter notebook Klasifikasi_Sampah.ipynb`.

## Cara Penggunaan (Usage)
1. **Preprocessing & EDA**: Jalankan sel pertama untuk import library dan verifikasi dataset. Lihat visualisasi distribusi dan contoh citra.
2. **Training Model**:
   - CNN: Gunakan `ImageDataGenerator` untuk augmentasi (rotation, flip, zoom). Train dengan 50 epochs.
   - Model Tradisional: Ekstrak fitur dari citra, train dengan `train_test_split(0.2)`.
3. **Evaluasi**: Lihat plot akurasi, confusion matrix, dan tabel perbandingan model.
4. **Simpan Model**: Model disimpan sebagai `.h5` (CNN) atau `.pkl` (RF). Load dengan:
   ```python
   model = tf.keras.models.load_model('cnn_model.h5')
   ```

Contoh inferensi pada gambar baru:
```python
from tensorflow.keras.preprocessing import image
img = image.load_img('path_to_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
prediction = model.predict(np.expand_dims(img_array, axis=0))
print("Recycle" if prediction[0] > 0.5 else "Non-Recycle")
```

## Struktur File
```
Klasifikasi_Sampah/
├── Klasifikasi_Sampah.ipynb       # Notebook utama
├── requirements.txt               # Dependencies
├── dataset/                       # (Opsional) Copy dataset lokal
│   ├── recycle/                   # 285 gambar
│   └── non-recycle/               # 114 gambar
├── models/                        # Output model
│   ├── cnn_model.h5
│   └── rf_model.pkl
└── README.md                      # Dokumen ini
```

## Hasil dan Temuan
- **Distribusi Dataset**: 71.4% recycle, 28.6% non-recycle (tidak seimbang → augmentasi membantu).
- **Performa Model**:
  | Model          | Test Accuracy | Precision (Recycle) | Recall (Recycle) | F1-Score (Recycle) |
  |----------------|---------------|---------------------|------------------|--------------------|
  | CNN (Keras)   | 83.54%       | 0.85                | 0.88             | 0.86               |
  | Random Forest | ~78.50%      | 0.80                | 0.82             | 0.81               |
  | SVM           | ~75.20%      | 0.77                | 0.79             | 0.78               |

- **Insights**:
  - CNN lebih baik dalam menangkap fitur spasial citra dibandingkan model tradisional.
  - Overfitting negatif pada CNN menunjukkan generalisasi baik, tapi validasi set kecil mungkin memengaruhi.
  - Tantangan: Dataset kecil (399 gambar) → augmentasi krusial untuk hindari overfitting.

## Langkah Selanjutnya (Next Steps)
- Tambah kelas multi (e.g., plastik, kertas, organik).
- Gunakan model pre-trained (Transfer Learning: ResNet50/VGG16) untuk akurasi >90%.
- Deploy ke web app (Streamlit/Flask) untuk deteksi real-time via webcam.
- Perbesar dataset dengan synthetic data (GAN) atau crowdsourcing.
- Optimasi: Hyperparameter tuning dengan GridSearchCV atau Keras Tuner.

## Kontribusi
Silakan fork repositori ini dan buat pull request untuk perbaikan. Laporkan issue jika ada bug atau saran dataset baru.

## Lisensi
MIT License - Gunakan bebas untuk tujuan edukasi/penelitian.
