import cv2
import csv
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# KONFIGURASI
# ============================================================

VIDEO_DIR = Path("videos")
RESULT_DIR = Path("results")

# Model MediaPipe Tasks
MODEL_PATH = Path(
    "models/face_detector.tflite"
)

# Sampling frame
# 1 = semua frame
# 5 = proses 1 dari setiap 5 frame
FRAME_SKIP = 5

# Minimum confidence deteksi wajah
MIN_DETECTION_CONFIDENCE = 0.5


# ============================================================
# FOLDER FRAME HASIL
# ============================================================

OBJECT_FRAME_DIR = (
    RESULT_DIR /
    "object_frames"
)

OBJECT_FRAME_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDASI MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(

        f"Model tidak ditemukan:\n"
        f"{MODEL_PATH.resolve()}\n\n"

        "Pastikan file "
        "face_detector.tflite "
        "berada di folder models."
    )


# ============================================================
# PERSIAPAN FOLDER
# ============================================================

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FACE DETECTOR
# MediaPipe Tasks API
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(
        MODEL_PATH
    )
)


options = vision.FaceDetectorOptions(

    base_options=base_options,

    min_detection_confidence=(
        MIN_DETECTION_CONFIDENCE
    )
)


detector = (
    vision.FaceDetector
    .create_from_options(
        options
    )
)


# ============================================================
# FUNGSI UNTUK MENGGAMBAR HASIL DETEKSI
# ============================================================

def draw_detections(
    frame,
    detections
):

    # Ukuran frame
    height, width = frame.shape[:2]


    # ========================================================
    # LOOP SETIAP WAJAH
    # ========================================================

    for index, detection in enumerate(
        detections,
        start=1
    ):

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        bbox = (
            detection.bounding_box
        )


        x = bbox.origin_x
        y = bbox.origin_y

        w = bbox.width
        h = bbox.height


        # ----------------------------------------------------
        # Pastikan koordinat berada
        # di dalam frame
        # ----------------------------------------------------

        x1 = max(
            0,
            x
        )

        y1 = max(
            0,
            y
        )

        x2 = min(
            width,
            x + w
        )

        y2 = min(
            height,
            y + h
        )


        # ----------------------------------------------------
        # Ambil confidence
        # ----------------------------------------------------

        confidence = 0.0

        if detection.categories:

            confidence = (
                detection.categories[0]
                .score
            )


        # ----------------------------------------------------
        # Gambar bounding box
        # ----------------------------------------------------

        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            (0, 255, 0),

            2
        )


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (

            f"FACE {index} "
            f"({confidence:.2f})"
        )


        # ----------------------------------------------------
        # Background label
        # ----------------------------------------------------

        label_size = (
            cv2.getTextSize(

                label,

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                2
            )[0]
        )


        label_width = (
            label_size[0] + 10
        )

        label_height = (
            label_size[1] + 10
        )


        label_y = max(
            y1,
            label_height
        )


        cv2.rectangle(

            frame,

            (
                x1,
                label_y - label_height
            ),

            (
                x1 + label_width,
                label_y
            ),

            (0, 255, 0),

            -1
        )


        # ----------------------------------------------------
        # Tulis label
        # ----------------------------------------------------

        cv2.putText(

            frame,

            label,

            (
                x1 + 5,
                label_y - 5
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 0, 0),

            2
        )


    return frame


# ============================================================
# FUNGSI MENYIMPAN FRAME HASIL
# ============================================================

def save_detection_frame(
    frame,
    detections,
    video_path,
    condition
):

    # --------------------------------------------------------
    # Folder berdasarkan kondisi
    # --------------------------------------------------------

    condition_folder = (
        OBJECT_FRAME_DIR /
        condition
    )


    condition_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Gambar bounding box
    # --------------------------------------------------------

    result_frame = frame.copy()


    result_frame = draw_detections(

        result_frame,

        detections
    )


    # --------------------------------------------------------
    # Tambahkan informasi kondisi
    # --------------------------------------------------------

    if condition == "0_FACE":

        title = (
            "OBJECT DETECTION: "
            "0 FACE"
        )


    elif condition == "1_FACE":

        title = (
            "OBJECT DETECTION: "
            "1 FACE"
        )


    else:

        title = (
            "OBJECT DETECTION: "
            ">1 FACE"
        )


    # --------------------------------------------------------
    # Background informasi
    # --------------------------------------------------------

    cv2.rectangle(

        result_frame,

        (10, 10),

        (350, 55),

        (0, 0, 0),

        -1
    )


    # --------------------------------------------------------
    # Tulis informasi
    # --------------------------------------------------------

    cv2.putText(

        result_frame,

        title,

        (20, 42),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (0, 255, 255),

        2
    )


    # --------------------------------------------------------
    # Nama file
    # --------------------------------------------------------

    output_image = (

        condition_folder /

        f"{video_path.stem}"
        f"_{condition.lower()}.jpg"
    )


    # --------------------------------------------------------
    # Simpan
    # --------------------------------------------------------

    cv2.imwrite(

        str(output_image),

        result_frame
    )


    return output_image


# ============================================================
# FUNGSI ANALISIS SATU VIDEO
# ============================================================

def analyze_video(
    video_path
):

    print(
        f"\n  Membuka video: "
        f"{video_path.name}"
    )


    # ========================================================
    # BUKA VIDEO
    # ========================================================

    cap = cv2.VideoCapture(

        str(video_path)
    )


    if not cap.isOpened():

        print(

            f"  [ERROR] Tidak dapat membuka: "
            f"{video_path.name}"
        )

        return None


    # ========================================================
    # INFORMASI VIDEO
    # ========================================================

    total_frames = int(

        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    fps = cap.get(

        cv2.CAP_PROP_FPS
    )


    if fps <= 0:

        fps = 30.0


    # ========================================================
    # COUNTER
    # ========================================================

    processed_frames = 0

    zero_face = 0

    one_face = 0

    multiple_face = 0

    frame_index = 0


    # ========================================================
    # PENANDA FRAME YANG SUDAH DISIMPAN
    # ========================================================

    saved_frames = {

        "0_FACE": False,

        "1_FACE": False,

        "MULTIPLE_FACE": False
    }


    # ========================================================
    # LOOP VIDEO
    # ========================================================

    while True:

        ret, frame = cap.read()


        if not ret:

            break


        frame_index += 1


        # ====================================================
        # SAMPLING FRAME
        # ====================================================

        if (
            frame_index %
            FRAME_SKIP
            != 0
        ):

            continue


        processed_frames += 1


        # ====================================================
        # BGR -> RGB
        # ====================================================

        rgb_frame = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # MEDIAPIPE IMAGE
        # ====================================================

        mp_image = mp.Image(

            image_format=(
                mp.ImageFormat.SRGB
            ),

            data=rgb_frame
        )


        # ====================================================
        # FACE DETECTION
        # ====================================================

        result = detector.detect(

            mp_image
        )


        # ====================================================
        # JUMLAH WAJAH
        # ====================================================

        detections = (
            result.detections
        )


        face_count = len(
            detections
        )


        # ====================================================
        # STATISTIK
        # ====================================================

        if face_count == 0:

            zero_face += 1

            condition = "0_FACE"


        elif face_count == 1:

            one_face += 1

            condition = "1_FACE"


        else:

            multiple_face += 1

            condition = (
                "MULTIPLE_FACE"
            )


        # ====================================================
        # SIMPAN FRAME CONTOH
        # ====================================================

        if not saved_frames[condition]:

            output_image = (
                save_detection_frame(

                    frame=frame.copy(),

                    detections=detections,

                    video_path=video_path,

                    condition=condition
                )
            )


            saved_frames[
                condition
            ] = True


            print(

                f"  Frame hasil "
                f"{condition} disimpan: "
                f"{output_image}"
            )


    # ========================================================
    # TUTUP VIDEO
    # ========================================================

    cap.release()


    # ========================================================
    # HITUNG PERSENTASE
    # ========================================================

    if processed_frames > 0:

        zero_percent = (

            zero_face /

            processed_frames *

            100
        )


        one_percent = (

            one_face /

            processed_frames *

            100
        )


        multiple_percent = (

            multiple_face /

            processed_frames *

            100
        )


    else:

        zero_percent = 0.0

        one_percent = 0.0

        multiple_percent = 0.0


    # ========================================================
    # HASIL
    # ========================================================

    return {

        "video":
            video_path.name,

        "total_frames":
            total_frames,

        "fps":
            round(
                fps,
                2
            ),

        "processed_frames":
            processed_frames,

        "zero_face":
            zero_face,

        "one_face":
            one_face,

        "multiple_face":
            multiple_face,

        "zero_face_percent":
            round(
                zero_percent,
                2
            ),

        "one_face_percent":
            round(
                one_percent,
                2
            ),

        "multiple_face_percent":
            round(
                multiple_percent,
                2
            )
    }


# ============================================================
# CARI VIDEO
# ============================================================

VIDEO_EXTENSIONS = {

    ".mp4",

    ".avi",

    ".mov",

    ".mkv",

    ".webm",

    ".m4v"
}


video_files = sorted(

    [

        file

        for file in VIDEO_DIR.iterdir()

        if (

            file.is_file()

            and

            file.suffix.lower()
            in VIDEO_EXTENSIONS
        )
    ]
)


# ============================================================
# VALIDASI VIDEO
# ============================================================

if not video_files:

    detector.close()


    raise SystemExit(

        "Tidak ada video ditemukan "

        f"di folder: "
        f"{VIDEO_DIR.resolve()}"
    )


# ============================================================
# INFORMASI PROGRAM
# ============================================================

print(
    "=" * 65
)

print(
    "FACE DETECTION - MEDIAPIPE TASKS"
)

print(
    "=" * 65
)

print(

    f"Jumlah video : "
    f"{len(video_files)}"
)

print(

    f"Model       : "
    f"{MODEL_PATH}"
)

print(

    f"Frame skip  : "
    f"{FRAME_SKIP}"
)

print(

    f"Confidence  : "
    f"{MIN_DETECTION_CONFIDENCE}"
)

print(

    f"Frame hasil : "
    f"{OBJECT_FRAME_DIR}"
)

print(
    "=" * 65
)


# ============================================================
# PROSES SEMUA VIDEO
# ============================================================

all_results = []


for number, video_path in enumerate(

    video_files,

    start=1
):

    print(

        f"\n[{number}/"
        f"{len(video_files)}] "

        f"Memproses: "
        f"{video_path.name}"
    )


    result = analyze_video(

        video_path
    )


    if result is None:

        continue


    all_results.append(
        result
    )


    # ========================================================
    # TAMPILKAN HASIL
    # ========================================================

    print(

        f"  0 wajah  : "
        f"{result['zero_face_percent']:.2f}%"
    )


    print(

        f"  1 wajah  : "
        f"{result['one_face_percent']:.2f}%"
    )


    print(

        f"  >1 wajah : "
        f"{result['multiple_face_percent']:.2f}%"
    )


# ============================================================
# SIMPAN CSV
# ============================================================

output_csv = (

    RESULT_DIR /

    "object_detection_results.csv"
)


fieldnames = [

    "video",

    "total_frames",

    "fps",

    "processed_frames",

    "zero_face",

    "one_face",

    "multiple_face",

    "zero_face_percent",

    "one_face_percent",

    "multiple_face_percent"
]


with open(

    output_csv,

    "w",

    newline="",

    encoding="utf-8"

) as file:


    writer = csv.DictWriter(

        file,

        fieldnames=fieldnames
    )


    writer.writeheader()


    writer.writerows(

        all_results
    )


# ============================================================
# SELESAI
# ============================================================

detector.close()


print()

print(
    "=" * 65
)

print(
    "SELESAI"
)

print(
    "=" * 65
)

print(

    f"Hasil CSV disimpan di:\n"
    f"{output_csv.resolve()}"
)

print()

print(

    f"Frame hasil disimpan di:\n"
    f"{OBJECT_FRAME_DIR.resolve()}"
)

print(
    "=" * 65
)