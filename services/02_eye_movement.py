from pathlib import Path
import cv2
import csv
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# KONFIGURASI
# ============================================================

VIDEO_DIR = Path("videos")
RESULT_DIR = Path("results")

MODEL_PATH = Path(
    "models/face_landmarker.task"
)

# Sampling frame
# 1 = semua frame
# 5 = proses 1 dari setiap 5 frame
FRAME_SKIP = 5


# ============================================================
# THRESHOLD GAZE
# ============================================================

# GeoGaze horizontal baseline
#
# X < 0.40  -> RIGHT
# X 0.40-0.60 -> CENTER
# X > 0.60  -> LEFT
#
# Catatan:
# Threshold horizontal ini menggunakan pendekatan
# berbasis GeoGaze.
#
RIGHT_THRESHOLD = 0.40
LEFT_THRESHOLD = 0.60


# Vertical extension
#
# Y < 0.40 -> UP
# Y 0.40-0.60 -> CENTER-Y
# Y > 0.60 -> DOWN
#
# Threshold vertical ini merupakan
# ekstensi untuk kebutuhan proyek.
#
UP_THRESHOLD = 0.40
DOWN_THRESHOLD = 0.60


# ============================================================
# VALIDASI RATIO
# ============================================================

MIN_RATIO = 0.0
MAX_RATIO = 1.0


# ============================================================
# LANDMARK INDEX
# MediaPipe Face Landmarker
# ============================================================

# ------------------------------------------------------------
# Mata kiri
# ------------------------------------------------------------

LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145


# ------------------------------------------------------------
# Mata kanan
# ------------------------------------------------------------

RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


# ------------------------------------------------------------
# Iris kiri
# ------------------------------------------------------------

LEFT_IRIS = [
    468,
    469,
    470,
    471,
    472
]


# ------------------------------------------------------------
# Iris kanan
# ------------------------------------------------------------

RIGHT_IRIS = [
    473,
    474,
    475,
    476,
    477
]


# ============================================================
# FOLDER HASIL
# ============================================================

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Folder untuk menyimpan frame contoh
GAZE_FRAME_DIR = (
    RESULT_DIR /
    "gaze_frames"
)

GAZE_FRAME_DIR.mkdir(
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
        "face_landmarker.task "
        "berada di folder models."
    )


# ============================================================
# FUNGSI
# MENGHITUNG RATIO HORIZONTAL
# ============================================================

def calculate_horizontal_ratio(
    landmarks,
    eye_left_index,
    eye_right_index,
    iris_indices
):

    # --------------------------------------------------------
    # Koordinat sudut mata
    # --------------------------------------------------------

    eye_left_x = (
        landmarks[eye_left_index].x
    )

    eye_right_x = (
        landmarks[eye_right_index].x
    )


    # --------------------------------------------------------
    # Karena posisi landmark bisa berbeda
    # --------------------------------------------------------

    eye_min_x = min(
        eye_left_x,
        eye_right_x
    )

    eye_max_x = max(
        eye_left_x,
        eye_right_x
    )


    eye_width = (
        eye_max_x -
        eye_min_x
    )


    # Hindari pembagian dengan nol
    if eye_width <= 0:

        return None


    # --------------------------------------------------------
    # Posisi tengah iris
    # --------------------------------------------------------

    iris_x = sum(
        landmarks[index].x
        for index in iris_indices
    ) / len(iris_indices)


    # --------------------------------------------------------
    # Normalisasi posisi iris
    # --------------------------------------------------------

    ratio = (
        iris_x -
        eye_min_x
    ) / eye_width


    return ratio


# ============================================================
# FUNGSI
# MENGHITUNG RATIO VERTIKAL
# ============================================================

def calculate_vertical_ratio(
    landmarks,
    eye_top_index,
    eye_bottom_index,
    iris_indices
):

    # --------------------------------------------------------
    # Koordinat mata
    # --------------------------------------------------------

    eye_top_y = (
        landmarks[eye_top_index].y
    )

    eye_bottom_y = (
        landmarks[eye_bottom_index].y
    )


    # --------------------------------------------------------
    # Posisi min dan max
    # --------------------------------------------------------

    eye_min_y = min(
        eye_top_y,
        eye_bottom_y
    )

    eye_max_y = max(
        eye_top_y,
        eye_bottom_y
    )


    eye_height = (
        eye_max_y -
        eye_min_y
    )


    # Hindari pembagian dengan nol
    if eye_height <= 0:

        return None


    # --------------------------------------------------------
    # Posisi tengah iris
    # --------------------------------------------------------

    iris_y = sum(
        landmarks[index].y
        for index in iris_indices
    ) / len(iris_indices)


    # --------------------------------------------------------
    # Normalisasi posisi iris
    # --------------------------------------------------------

    ratio = (
        iris_y -
        eye_min_y
    ) / eye_height


    return ratio


# ============================================================
# VALIDASI NILAI RATIO
# ============================================================

def is_valid_ratio(value):

    if value is None:
        return False

    return (
        MIN_RATIO <= value <= MAX_RATIO
    )


# ============================================================
# ANALISIS GAZE
# ============================================================

def analyze_gaze(
    face_landmarks
):

    # ========================================================
    # RATIO MATA KIRI
    # ========================================================

    left_x = calculate_horizontal_ratio(
        face_landmarks,
        LEFT_EYE_LEFT,
        LEFT_EYE_RIGHT,
        LEFT_IRIS
    )

    left_y = calculate_vertical_ratio(
        face_landmarks,
        LEFT_EYE_TOP,
        LEFT_EYE_BOTTOM,
        LEFT_IRIS
    )


    # ========================================================
    # RATIO MATA KANAN
    # ========================================================

    right_x = calculate_horizontal_ratio(
        face_landmarks,
        RIGHT_EYE_LEFT,
        RIGHT_EYE_RIGHT,
        RIGHT_IRIS
    )

    right_y = calculate_vertical_ratio(
        face_landmarks,
        RIGHT_EYE_TOP,
        RIGHT_EYE_BOTTOM,
        RIGHT_IRIS
    )


    # ========================================================
    # VALIDASI RATIO INDIVIDUAL
    # ========================================================

    ratios = [
        left_x,
        left_y,
        right_x,
        right_y
    ]


    if not all(
        is_valid_ratio(value)
        for value in ratios
    ):

        return (
            "UNKNOWN",
            None,
            None,
            False
        )


    # ========================================================
    # RATA-RATA KEDUA MATA
    # ========================================================

    gaze_x = (
        left_x +
        right_x
    ) / 2


    gaze_y = (
        left_y +
        right_y
    ) / 2


    # ========================================================
    # VALIDASI RATIO HASIL
    # ========================================================

    if not is_valid_ratio(gaze_x):

        return (
            "UNKNOWN",
            None,
            None,
            False
        )


    if not is_valid_ratio(gaze_y):

        return (
            "UNKNOWN",
            None,
            None,
            False
        )


    # ========================================================
    # KLASIFIKASI HORIZONTAL
    # ========================================================

    if gaze_x < RIGHT_THRESHOLD:

        gaze = "RIGHT"


    elif gaze_x > LEFT_THRESHOLD:

        gaze = "LEFT"


    # ========================================================
    # KLASIFIKASI VERTIKAL
    # ========================================================

    elif gaze_y < UP_THRESHOLD:

        gaze = "UP"


    elif gaze_y > DOWN_THRESHOLD:

        gaze = "DOWN"


    # ========================================================
    # CENTER
    # ========================================================

    else:

        gaze = "CENTER"


    return (
        gaze,
        gaze_x,
        gaze_y,
        True
    )


# ============================================================
# GAMBAR LANDMARK MATA DAN IRIS
# ============================================================

def draw_eye_landmarks(
    frame,
    face_landmarks,
    gaze,
    gaze_x,
    gaze_y
):

    height, width = frame.shape[:2]


    # ========================================================
    # LANDMARK YANG AKAN DIGAMBAR
    # ========================================================

    landmark_indices = [

        # Mata kiri
        LEFT_EYE_LEFT,
        LEFT_EYE_RIGHT,
        LEFT_EYE_TOP,
        LEFT_EYE_BOTTOM,

        # Mata kanan
        RIGHT_EYE_LEFT,
        RIGHT_EYE_RIGHT,
        RIGHT_EYE_TOP,
        RIGHT_EYE_BOTTOM,

        # Iris kiri
        *LEFT_IRIS,

        # Iris kanan
        *RIGHT_IRIS
    ]


    # ========================================================
    # GAMBAR LANDMARK
    # ========================================================

    for index in landmark_indices:

        landmark = (
            face_landmarks[index]
        )


        x = int(
            landmark.x *
            width
        )

        y = int(
            landmark.y *
            height
        )


        # Pastikan titik masih berada
        # di dalam gambar

        if (
            x < 0
            or x >= width
            or y < 0
            or y >= height
        ):

            continue


        cv2.circle(

            frame,

            (x, y),

            3,

            (0, 255, 255),

            -1
        )


    # ========================================================
    # INFORMASI GAZE
    # ========================================================

    cv2.rectangle(

        frame,

        (10, 10),

        (310, 115),

        (0, 0, 0),

        -1
    )


    # --------------------------------------------------------
    # Label Gaze
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"GAZE: {gaze}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 255, 255),

        2
    )


    # --------------------------------------------------------
    # Ratio X
    # --------------------------------------------------------

    if gaze_x is not None:

        cv2.putText(

            frame,

            f"X: {gaze_x:.3f}",

            (20, 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            2
        )


    # --------------------------------------------------------
    # Ratio Y
    # --------------------------------------------------------

    if gaze_y is not None:

        cv2.putText(

            frame,

            f"Y: {gaze_y:.3f}",

            (20, 98),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            2
        )


    return frame


# ============================================================
# SIMPAN FRAME GAZE
# ============================================================

def save_gaze_frame(
    frame,
    video_path,
    gaze,
    face_landmarks=None,
    gaze_x=None,
    gaze_y=None
):

    # --------------------------------------------------------
    # Folder berdasarkan arah
    # --------------------------------------------------------

    gaze_folder = (
        GAZE_FRAME_DIR /
        gaze
    )


    gaze_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Nama file
    # --------------------------------------------------------

    output_image = (

        gaze_folder /

        f"{video_path.stem}"
        f"_{gaze.lower()}.jpg"
    )


    # --------------------------------------------------------
    # Jika ada landmark
    # --------------------------------------------------------

    if face_landmarks is not None:

        frame = draw_eye_landmarks(

            frame,

            face_landmarks,

            gaze,

            gaze_x,

            gaze_y
        )


    else:

        # ----------------------------------------------------
        # UNKNOWN karena tidak ada wajah
        # ----------------------------------------------------

        cv2.rectangle(

            frame,

            (10, 10),

            (350, 60),

            (0, 0, 0),

            -1
        )


        cv2.putText(

            frame,

            "GAZE: UNKNOWN - NO FACE",

            (20, 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 255),

            2
        )


    # --------------------------------------------------------
    # Simpan gambar
    # --------------------------------------------------------

    cv2.imwrite(

        str(output_image),

        frame
    )


    return output_image


# ============================================================
# ANALISIS SATU VIDEO
# ============================================================

def analyze_video(
    video_path
):

    print(
        f"\n  Membuka video: "
        f"{video_path.name}"
    )


    # ========================================================
    # OPEN VIDEO
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

    frames_analyzed = 0

    face_detected = 0

    invalid_ratio = 0

    center = 0
    left = 0
    right = 0
    up = 0
    down = 0
    unknown = 0


    # ========================================================
    # PENANDA FRAME CONTOH
    # ========================================================

    saved_gaze_frames = {

        "CENTER": False,

        "LEFT": False,

        "RIGHT": False,

        "UP": False,

        "DOWN": False,

        "UNKNOWN": False
    }


    # ========================================================
    # TIMELINE
    # ========================================================

    timeline_rows = []


    # ========================================================
    # FRAME INDEX
    # ========================================================

    frame_index = 0


    # ========================================================
    # FACE LANDMARKER
    # ========================================================

    base_options = (
        python.BaseOptions(
            model_asset_path=str(
                MODEL_PATH
            )
        )
    )


    options = (
        vision.FaceLandmarkerOptions(

            base_options=base_options,

            running_mode=(
                vision.RunningMode.VIDEO
            ),

            num_faces=1,

            min_face_detection_confidence=0.5,

            min_face_presence_confidence=0.5,

            min_tracking_confidence=0.5
        )
    )


    landmarker = (
        vision.FaceLandmarker
        .create_from_options(
            options
        )
    )


    # ========================================================
    # LOOP VIDEO
    # ========================================================

    while True:

        ret, frame = cap.read()


        if not ret:

            break


        frame_index += 1


        # ====================================================
        # SAMPLING
        # ====================================================

        if (
            frame_index %
            FRAME_SKIP
            != 0
        ):

            continue


        frames_analyzed += 1


        # ====================================================
        # TIMESTAMP
        # ====================================================

        time_sec = (
            frame_index /
            fps
        )


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
        # TIMESTAMP MILISECOND
        # ====================================================

        timestamp_ms = int(

            (frame_index / fps)
            * 1000
        )


        # ====================================================
        # FACE LANDMARK
        # ====================================================

        result = (
            landmarker.detect_for_video(

                mp_image,

                timestamp_ms
            )
        )


        # ====================================================
        # TIDAK ADA WAJAH
        # ====================================================

        if not result.face_landmarks:

            gaze = "UNKNOWN"

            gaze_x = None

            gaze_y = None

            ratio_status = (
                "NO_FACE"
            )

            unknown += 1


            # ------------------------------------------------
            # Simpan frame UNKNOWN
            # ------------------------------------------------

            if not saved_gaze_frames[
                "UNKNOWN"
            ]:

                output_image = (
                    save_gaze_frame(

                        frame.copy(),

                        video_path,

                        "UNKNOWN"
                    )
                )


                saved_gaze_frames[
                    "UNKNOWN"
                ] = True


                print(

                    f"  Frame contoh "
                    f"UNKNOWN disimpan: "
                    f"{output_image}"
                )


        else:

            # =================================================
            # WAJAH TERDETEKSI
            # =================================================

            face_detected += 1


            # Ambil landmark wajah pertama

            face_landmarks = (
                result.face_landmarks[0]
            )


            # =================================================
            # ANALISIS GAZE
            # =================================================

            (
                gaze,
                gaze_x,
                gaze_y,
                valid
            ) = analyze_gaze(
                face_landmarks
            )


            # =================================================
            # VALID / INVALID
            # =================================================

            if not valid:

                ratio_status = (
                    "INVALID_RATIO"
                )

                invalid_ratio += 1

                unknown += 1


                # ------------------------------------------------
                # Simpan frame UNKNOWN
                # ------------------------------------------------

                if not saved_gaze_frames[
                    "UNKNOWN"
                ]:

                    output_image = (
                        save_gaze_frame(

                            frame.copy(),

                            video_path,

                            "UNKNOWN",

                            face_landmarks,

                            gaze_x,

                            gaze_y
                        )
                    )


                    saved_gaze_frames[
                        "UNKNOWN"
                    ] = True


                    print(

                        f"  Frame contoh "
                        f"UNKNOWN disimpan: "
                        f"{output_image}"
                    )


            else:

                ratio_status = "VALID"


                # =================================================
                # HITUNG GAZE
                # =================================================

                if gaze == "CENTER":

                    center += 1


                elif gaze == "LEFT":

                    left += 1


                elif gaze == "RIGHT":

                    right += 1


                elif gaze == "UP":

                    up += 1


                elif gaze == "DOWN":

                    down += 1


                else:

                    unknown += 1


                # =================================================
                # SIMPAN FRAME CONTOH
                # =================================================

                if (

                    gaze in saved_gaze_frames

                    and not
                    saved_gaze_frames[gaze]

                ):

                    output_image = (
                        save_gaze_frame(

                            frame.copy(),

                            video_path,

                            gaze,

                            face_landmarks,

                            gaze_x,

                            gaze_y
                        )
                    )


                    saved_gaze_frames[
                        gaze
                    ] = True


                    print(

                        f"  Frame contoh "
                        f"{gaze} disimpan: "
                        f"{output_image}"
                    )


        # ========================================================
        # TIMELINE
        # ========================================================

        timeline_rows.append({

            "video":
                video_path.name,

            "frame":
                frame_index,

            "time_sec":
                round(
                    time_sec,
                    3
                ),

            "gaze":
                gaze,

            "gaze_x":
                (
                    round(
                        gaze_x,
                        4
                    )
                    if gaze_x is not None
                    else ""
                ),

            "gaze_y":
                (
                    round(
                        gaze_y,
                        4
                    )
                    if gaze_y is not None
                    else ""
                ),

            "ratio_status":
                ratio_status
        })


    # ========================================================
    # TUTUP
    # ========================================================

    cap.release()

    landmarker.close()


    # ========================================================
    # PERSENTASE
    # ========================================================

    if frames_analyzed > 0:

        face_detected_percent = (
            face_detected /
            frames_analyzed *
            100
        )

        invalid_ratio_percent = (
            invalid_ratio /
            frames_analyzed *
            100
        )

        center_percent = (
            center /
            frames_analyzed *
            100
        )

        left_percent = (
            left /
            frames_analyzed *
            100
        )

        right_percent = (
            right /
            frames_analyzed *
            100
        )

        up_percent = (
            up /
            frames_analyzed *
            100
        )

        down_percent = (
            down /
            frames_analyzed *
            100
        )

        unknown_percent = (
            unknown /
            frames_analyzed *
            100
        )


    else:

        face_detected_percent = 0.0

        invalid_ratio_percent = 0.0

        center_percent = 0.0

        left_percent = 0.0

        right_percent = 0.0

        up_percent = 0.0

        down_percent = 0.0

        unknown_percent = 0.0


    # ========================================================
    # RETURN SUMMARY
    # ========================================================

    summary = {

        "video":
            video_path.name,

        "total_frames":
            total_frames,

        "fps":
            round(
                fps,
                2
            ),

        "frames_analyzed":
            frames_analyzed,

        "face_detected_percent":
            round(
                face_detected_percent,
                2
            ),

        "invalid_ratio_percent":
            round(
                invalid_ratio_percent,
                2
            ),

        "center_percent":
            round(
                center_percent,
                2
            ),

        "left_percent":
            round(
                left_percent,
                2
            ),

        "right_percent":
            round(
                right_percent,
                2
            ),

        "up_percent":
            round(
                up_percent,
                2
            ),

        "down_percent":
            round(
                down_percent,
                2
            ),

        "unknown_percent":
            round(
                unknown_percent,
                2
            )
    }


    return (
        summary,
        timeline_rows
    )


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

    raise SystemExit(

        "Tidak ada video ditemukan "
        f"di folder: "
        f"{VIDEO_DIR.resolve()}"
    )


# ============================================================
# INFORMASI PROGRAM
# ============================================================

print()
print("=" * 70)

print(
    "EYE MOVEMENT ANALYSIS"
)

print(
    "MEDIAPIPE FACE LANDMARKER"
)

print(
    "GEOGAZE + VERTICAL EXTENSION"
)

print("=" * 70)

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
    f"GeoGaze X   : "
    f"RIGHT < {RIGHT_THRESHOLD:.2f} | "
    f"CENTER {RIGHT_THRESHOLD:.2f}-"
    f"{LEFT_THRESHOLD:.2f} | "
    f"LEFT > {LEFT_THRESHOLD:.2f}"
)

print(
    f"Vertical Y  : "
    f"UP < {UP_THRESHOLD:.2f} | "
    f"DOWN > {DOWN_THRESHOLD:.2f}"
)

print(
    f"Valid ratio : "
    f"{MIN_RATIO:.1f} - "
    f"{MAX_RATIO:.1f}"
)

print(
    f"Frame gaze  : "
    f"{GAZE_FRAME_DIR}"
)

print("=" * 70)


# ============================================================
# PROSES SEMUA VIDEO
# ============================================================

all_results = []

all_timeline = []


for number, video_path in enumerate(

    video_files,

    start=1
):

    print()

    print(

        f"[{number}/"
        f"{len(video_files)}] "
        f"Memproses: "
        f"{video_path.name}"
    )


    result = analyze_video(

        video_path
    )


    if result is None:

        continue


    summary, timeline = result


    all_results.append(
        summary
    )


    all_timeline.extend(
        timeline
    )


    # ========================================================
    # TAMPILKAN HASIL
    # ========================================================

    print()

    print(
        f"  Frame analyzed    : "
        f"{summary['frames_analyzed']}"
    )

    print(
        f"  Face detected     : "
        f"{summary['face_detected_percent']:.2f}%"
    )

    print(
        f"  Invalid ratio     : "
        f"{summary['invalid_ratio_percent']:.2f}%"
    )

    print(
        f"  CENTER            : "
        f"{summary['center_percent']:.2f}%"
    )

    print(
        f"  LEFT              : "
        f"{summary['left_percent']:.2f}%"
    )

    print(
        f"  RIGHT             : "
        f"{summary['right_percent']:.2f}%"
    )

    print(
        f"  UP                : "
        f"{summary['up_percent']:.2f}%"
    )

    print(
        f"  DOWN              : "
        f"{summary['down_percent']:.2f}%"
    )

    print(
        f"  UNKNOWN           : "
        f"{summary['unknown_percent']:.2f}%"
    )


# ============================================================
# SIMPAN SUMMARY CSV
# ============================================================

summary_csv = (

    RESULT_DIR /

    "eye_movement_results.csv"
)


summary_fieldnames = [

    "video",

    "total_frames",

    "fps",

    "frames_analyzed",

    "face_detected_percent",

    "invalid_ratio_percent",

    "center_percent",

    "left_percent",

    "right_percent",

    "up_percent",

    "down_percent",

    "unknown_percent"
]


with open(

    summary_csv,

    "w",

    newline="",

    encoding="utf-8"

) as file:

    writer = csv.DictWriter(

        file,

        fieldnames=
            summary_fieldnames
    )


    writer.writeheader()


    writer.writerows(

        all_results
    )


# ============================================================
# SIMPAN TIMELINE CSV
# ============================================================

timeline_csv = (

    RESULT_DIR /

    "eye_movement_timeline.csv"
)


timeline_fieldnames = [

    "video",

    "frame",

    "time_sec",

    "gaze",

    "gaze_x",

    "gaze_y",

    "ratio_status"
]


with open(

    timeline_csv,

    "w",

    newline="",

    encoding="utf-8"

) as file:

    writer = csv.DictWriter(

        file,

        fieldnames=
            timeline_fieldnames
    )


    writer.writeheader()


    writer.writerows(

        all_timeline
    )


# ============================================================
# SELESAI
# ============================================================

print()

print("=" * 70)

print(
    "SEMUA VIDEO SELESAI "
    "DIANALISIS"
)

print("=" * 70)

print(
    f"Summary : "
    f"{summary_csv.resolve()}"
)

print(
    f"Timeline: "
    f"{timeline_csv.resolve()}"
)

print(
    f"Frame   : "
    f"{GAZE_FRAME_DIR.resolve()}"
)

print("=" * 70)