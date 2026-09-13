# Python 3.12, bukan 3.14 seperti di host: coverage wheel untuk mediapipe,
# hdbscan, dan scikit-learn paling lengkap di 3.12.
FROM python:3.12-slim

# ffmpeg          -> ekstraksi audio di services/media.py
# libgl1, libglib -> dibutuhkan opencv & mediapipe
# libsndfile1     -> dibutuhkan soundfile/librosa
# build-essential -> jaga-jaga kalau hdbscan harus dikompilasi dari source
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements disalin duluan supaya layer pip tidak ikut ter-rebuild
# setiap kali kode berubah
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# timeout-keep-alive dinaikkan karena satu request bisa jalan beberapa menit
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--timeout-keep-alive", "600"]
