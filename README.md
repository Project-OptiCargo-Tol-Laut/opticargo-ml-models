# 🧠 OptiCargo ML Models Service

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

Repositori ini berisi *backend service* berbasis **FastAPI** yang bertugas melayani prediksi analitik dan Machine Learning untuk ekosistem OptiCargo (Tol Laut). Service ini bertindak sebagai "Otak Kalkulasi" yang akan dipanggil oleh AI Agent secara *real-time* melalui HTTP Request.

Saat ini, model beroperasi dalam mode **MVP (Heuristic)** dan dirancang dengan arsitektur modular agar dapat dengan mudah diganti ke model ML sesungguhnya di fase pengembangan berikutnya.

---

## ⚠️ Prasyarat Penting (Wajib Dibaca)

Service ini sangat bergantung pada skema data (*Pydantic Models*) yang didefinisikan secara terpusat. 
**Anda WAJIB menginstal repositori `opticargo-shared` secara lokal terlebih dahulu** sebelum menjalankan service ini. 

Pastikan struktur direktori lokal Anda seperti ini:
```text
📦 OptiCargo-Tol-Laut-Project
 ┣ 📂 opticargo-shared     <-- Harus diinstal terlebih dahulu
 ┗ 📂 opticargo-ml-models  <-- Repositori ini
```


## 🚀 Cara Instalasi dan Menjalankan Service

### 1. Persiapkan Environment
Pastikan Anda menggunakan Python 3.10 atau yang lebih baru. Buat dan aktifkan virtual environment:
```bash
pip install -e .

python -m venv venv

# Pengguna Windows:
venv\Scripts\activate

# Pengguna Mac/Linux:
source venv/bin/activate
```

### 2. Install Repositori opticargo-shared
Arahkan terminal ke folder opticargo-shared lokal Anda, lalu install dalam editable mode:

```bash
cd ../opticargo-shared
pip install -e .
```

### 3. Install Dependensi ML Models
Kembali ke folder opticargo-ml-models, lalu install semua kebutuhan library:

```bash
cd ../opticargo-ml-models
pip install -r requirements.txt
```

### 4. Konfigurasi Environment Variables
Salin file template environment:

```bash
# Pengguna Windows:
copy .env.example .env

# Pengguna Mac/Linux:
cp .env.example .env
```

### 5. Jalankan Server FastAPI

Jalankan aplikasi menggunakan Uvicorn:
```bash
uvicorn serving.main:app --reload
```

## 📖 Cara Menggunakan API (Untuk Tim AI Agent)
Tim Agent dapat melihat dan mencoba dokumentasi interaktif (Swagger UI) dengan membuka browser ke:
👉 http://127.0.0.1:8000/docs