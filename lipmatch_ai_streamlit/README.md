# LipMatch AI - Streamlit Prototype

Aplikasi satu halaman untuk mengunggah foto wajah, mengambil sampel warna dari area pipi dan dahi, mengestimasi kategori warna Warm/Cool/Neutral, lalu menampilkan sembilan rekomendasi shade dari dataset lipstik Indonesia.

## Struktur folder

```text
lipmatch_ai_streamlit/
├── app.py
├── dataset.csv
├── requirements.txt
└── README.md
```

## Menjalankan di Windows

Disarankan memakai Python 3.11.

1. Ekstrak ZIP ini.
2. Buka folder `lipmatch_ai_streamlit` di VS Code.
3. Buka terminal pada folder tersebut.
4. Buat virtual environment:

   ```bash
   python -m venv .venv
   ```

5. Aktifkan environment:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   Jika menggunakan Command Prompt:

   ```bat
   .venv\Scripts\activate.bat
   ```

6. Install dependency:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Jalankan aplikasi:

   ```bash
   streamlit run app.py
   ```

Streamlit akan menampilkan alamat lokal di terminal, biasanya `http://localhost:8501`.

## Cara kerja prototype

1. MediaPipe Face Landmarker mendeteksi wajah.
2. Aplikasi mengambil median piksel pada tiga area sampel: pipi kiri, pipi kanan, dan dahi.
3. Aplikasi menghitung warna median dalam RGB, HEX, dan CIELAB.
4. Kategori Warm/Cool/Neutral diperkirakan menggunakan aturan heuristik awal yang digunakan pada notebook.
5. Warna HEX tiap lipstik pada `dataset.csv` dikonversi menjadi vektor RGB.
6. Cosine Similarity menghitung kemiripan arah antara vektor RGB warna kulit dan vektor RGB shade. Aplikasi memilih tiga kandidat teratas pada masing-masing kelompok Warm, Cool, dan Neutral.

## Catatan penting

- Kategori Warm/Cool/Neutral merupakan estimasi prototype. Foto dari kamera tidak dapat memastikan undertone biologis secara definitif.
- Cosine Similarity RGB adalah metode pencocokan komputasional sesuai tujuan prototype, tetapi bukan metrik persepsi warna yang tervalidasi untuk menentukan shade paling cocok. Nilai skor bukan probabilitas.
- Kategori shade di dataset ditentukan secara heuristik dari nama dan deskripsi produk. Validasi dengan label resmi brand atau ahli makeup akan meningkatkan kualitas hasil.
- Gunakan pencahayaan netral, foto frontal, dan hindari filter serta makeup tebal pada area pipi/dahi.
- Foto tidak ditulis ke disk oleh aplikasi ini. Namun, foto tetap diproses dalam memori selama sesi Streamlit berjalan.
- Model `face_landmarker.task` akan diunduh otomatis pada penggunaan pertama. Koneksi internet diperlukan jika model belum tersedia secara lokal.

## Deploy ke Streamlit Community Cloud

Upload `app.py`, `dataset.csv`, `requirements.txt`, dan `README.md` ke satu repository GitHub. Buat aplikasi baru di Streamlit Community Cloud dan tetapkan `app.py` sebagai main file. Gunakan Python 3.11 jika pilihan runtime tersedia.
