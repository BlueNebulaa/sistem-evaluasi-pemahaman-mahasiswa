# Sistem Evaluasi Pemahaman Mahasiswa

Aplikasi yang menilai pemahaman mahasiswa dari rekaman video wawancara teknis.
Satu video diunggah, sistem menjalankan seluruh analisis, lalu mengembalikan satu
JSON berisi skor jawaban sekaligus indikator kecurangan.

Dibangun sebagai MVP untuk keperluan tugas kuliah.

## Apa yang dianalisis

Dari satu video, sistem menghasilkan empat kelompok informasi:

| Analisis | Isi | Sumber |
|---|---|---|
| **Transkripsi** | Teks lengkap + segmen berwaktu | audio |
| **Penilaian** | Skor 0–4 per pertanyaan + justifikasi | transkrip + LLM |
| **Non-verbal** | Kecepatan bicara, jeda, stabilitas volume, filler word | audio |
| **Deteksi kecurangan** | Arah pandangan mata, jumlah pembicara | video + audio |

## Alur pipeline

```
                         ┌─ Transkripsi ──→ Penilaian (LLM) ─┐
                         │                                   │
 video.mp4 ─→ Ekstraksi ─┼─ Analisis non-verbal ─────────────┤
                audio                                        ├─→ JSON
                          ─ Object Detection             ────┤
                                                             │
 video.mp4 ─→ Eye tracking (arah pandangan) ─────────────────┘
```

Semua node berjalan berurutan. Kalau satu node gagal, node lain tetap jalan dan
hasilnya tetap keluar — node yang gagal muncul sebagai `{"error": "..."}` di JSON.
Ini disengaja: kegagalan deteksi wajah di menit ketiga tidak boleh membatalkan
transkrip dan skor yang sudah berhasil dihitung.

## Struktur project

```
.
├── app.py                  # FastAPI — titik masuk aplikasi
├── pipeline.py             # urutan pemanggilan semua node analisis
├── config.py               # konfigurasi terpusat, dibaca dari .env
├── rubrics.py              # 5 pertanyaan + rubrik penilaian
├── services/
│   ├── media.py            # mp4 → wav 16kHz (ffmpeg)
│   ├── transcription.py    # speech-to-text (faster-whisper)
│   ├── penilaian.py        # penilaian jawaban (Gemini)
│   ├── nonverbal.py        # WPM, jeda, volume, filler word (librosa)
│   ├── eye_tracking.py     # arah pandangan mata (mediapipe + opencv)
│   └── object_detection.py     
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

Prinsipnya: setiap file di `services/` berisi satu analisis dan tidak tahu-menahu
soal pipeline maupun HTTP. `pipeline.py` yang merangkainya, `app.py` yang mengurus
HTTP. Menambah analisis baru cukup tambah satu file di `services/` lalu daftarkan
satu baris di `pipeline.py`.

## Prasyarat

- Docker Desktop (cara yang disarankan), **atau** Python 3.12 + ffmpeg
- API key Google Gemini — bisa diambil gratis di <https://aistudio.google.com/apikey>

## Menjalankan dengan Docker

```bash
cp .env.example .env
# buka .env, isi GEMINI_API_KEY

docker compose up --build
```

Build pertama memakan waktu cukup lama karena mengunduh mediapipe dan dependensi
audio. Setelah selesai, buka:

**<http://localhost:8000/docs>**

Halaman itu menyediakan form upload langsung dari browser — tidak perlu Postman
atau curl untuk mencoba.

Menghentikan: `Ctrl+C`, lalu `docker compose down`.

## Menjalankan tanpa Docker

```bash
python -m venv venv
venv/Scripts/activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
cp .env.example .env           # lalu isi GEMINI_API_KEY

uvicorn app:app --reload --timeout-keep-alive 600
```

ffmpeg harus tersedia di PATH. Cek dengan `ffmpeg -version`; kalau belum ada, di
Windows bisa lewat `winget install ffmpeg`.

## Endpoint

| Method | Path | Keterangan |
|---|---|---|
| `GET` | `/health` | Cek aplikasi hidup |
| `POST` | `/evaluate` | Upload video, balas JSON hasil analisis |
| `GET` | `/docs` | Dokumentasi interaktif + form upload |

Contoh lewat curl:

```bash
curl -X POST http://localhost:8000/evaluate -F "file=@wawancara.mp4"
```

## Contoh output

```json
{
  "duration_sec": 182.5,
  "transcription": {
    "text": "I used transfer learning on MobileNet ...",
    "segments": [
      { "start": 0.0, "end": 3.0, "text": "...", "speaker": 0 }
    ],
    "duration": 182.5
  },
  "nonverbal": {
    "wpm": 132.5,
    "tempo_label": "Normal",
    "total_silence_sec": 21.4,
    "pause_label": "Normal",
    "volume_label": "Stable",
    "filler_count": 3,
    "filler_label": "Controlled"
  },
  "scoring": {
    "items": [
      { "no": 1, "score": 3, "justification": "Menyebut tantangan spesifik ..." }
    ],
    "total": 14,
    "max_total": 20
  },
  "cheating_detection": {
    "eye_tracking": {
      "efs_percent": 81.2,
      "edp_percent": 74.0,
      "final_confidence": 78.2,
      "status": "Valid"
    },
    "diarization": {
      "num_speakers": 2,
      "avg_similarity": 0.8731,
      "confidence_score": 81
    }
  }
}
```

Kalau ada node yang gagal, bagian itu berisi `{"error": "..."}` sementara bagian
lain tetap normal.

## Konfigurasi

Semua diatur lewat `.env`:

| Variabel | Default | Keterangan |
|---|---|---|
| `GEMINI_API_KEY` | — | **Wajib.** Tanpa ini, node penilaian error |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model penilai |
| `WHISPER_MODEL` | `small` | `tiny` / `base` / `small` / `medium`. Makin besar makin akurat tapi makin lambat |
| `WHISPER_DEVICE` | `cpu` | Isi `cuda` kalau punya GPU NVIDIA |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` untuk CPU, `float16` untuk GPU |
| `EYE_MAX_FRAMES` | `1000` | Batas frame yang dianalisis (lihat keterbatasan di bawah) |
| `EYE_FRAME_SKIP` | `2` | Analisis tiap N frame |

Mengubah `.env` cukup restart container, tidak perlu rebuild.

## Mengubah soal dan rubrik

Semua ada di `rubrics.py` sebagai list `QUESTIONS`. Tambah atau ubah entry di situ
— `services/penilaian.py` otomatis mengikuti, tidak ada kode yang perlu disesuaikan.

## Keterbatasan yang perlu diketahui

Beberapa hal sengaja disederhanakan karena ini MVP. Pahami sebelum demo:

1. **Proses berjalan sinkron.** Satu request menunggu sampai seluruh analisis
   selesai — bisa beberapa menit untuk video panjang. Gunakan video pendek (2–3
   menit) saat demo. Untuk produksi, pola yang benar adalah antrian job dengan
   endpoint status terpisah.

2. **Eye tracking hanya menganalisis frame awal.** Dengan default
   `EYE_MAX_FRAMES=1000` dan `EYE_FRAME_SKIP=2`, pada video 30fps hanya sekitar
   **33 detik pertama** yang dianalisis. Naikkan nilainya untuk video lebih
   panjang, dengan konsekuensi waktu proses bertambah.

3. **Object detection belum diimplementasikan.** Node ini ada di rancangan awal
   (deteksi HP atau orang kedua di frame) tapi belum dikerjakan.

4. **Pemetaan jawaban ke pertanyaan dilakukan oleh LLM.** Transkrip dikirim utuh
   dan LLM yang mencari bagian mana menjawab pertanyaan mana. Sederhana dan murah,
   tapi akurasinya belum diukur. Kalau hasilnya meleset, alternatifnya memecah
   transkrip per pertanyaan lebih dulu.

5. **Belum diuji dengan video sungguhan.** Rangkaian pipeline sudah diverifikasi,
   tapi jalur end-to-end dengan file mp4 nyata belum pernah dijalankan.

## Troubleshooting

**`ffmpeg tidak ditemukan di PATH`**
Pakai Docker, atau install ffmpeg manual dan pastikan masuk PATH.

**`GEMINI_API_KEY belum diisi di .env`**
Salin `.env.example` ke `.env` lalu isi. Error ini sengaja dimunculkan sebagai
pesan jelas, bukan disamarkan jadi skor 0.

**Build Docker gagal di `mediapipe`**
Kemungkinan wheel untuk versi yang di-pin tidak tersedia di platform Anda. Ganti
pin di `requirements.txt` menjadi `mediapipe==0.10.14` lalu build ulang.

**Request timeout / koneksi putus**
Video terlalu panjang untuk mode sinkron. Perpendek videonya, atau turunkan
`WHISPER_MODEL` ke `base`.
