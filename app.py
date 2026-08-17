import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO

import threading
import socket
import time
import math


app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

HOST = "0.0.0.0"
PORT = 5000

MODEL_NAME = "yolo11s.pt"

# 16:9
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# Camera upload FPS
PHONE_FPS = 15

# Laptop display FPS
OUTPUT_FPS = 20

# YOLO processing interval
# YOLO doesn't need to process every camera frame.
DETECTION_INTERVAL = 0.08

# JPEG quality
JPEG_QUALITY = 70


# ============================================================
# SPEED CALIBRATION
# ============================================================

# IMPORTANT:
# Distance between CALIBRATION_LINE_A and B in real life.
#
# Example:
# If you physically measure the road distance between
# the two virtual lines as 10 meters:
#
# CALIBRATION_DISTANCE_METERS = 10
#
CALIBRATION_DISTANCE_METERS = 10.0


# Speed thresholds
NORMAL_SPEED = 40
MODERATE_SPEED = 60


# Maximum accepted speed
MAX_SPEED = 200


# ============================================================
# VEHICLE CLASSES - COCO
# ============================================================

VEHICLE_CLASSES = {

    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"

}


# ============================================================
# CONFIDENCE
# ============================================================

CLASS_CONFIDENCE = {

    1: 0.18,   # bicycle
    2: 0.25,   # car
    3: 0.15,   # motorcycle
    5: 0.25,   # bus
    7: 0.25    # truck

}


# ============================================================
# GLOBAL CAMERA DATA
# ============================================================

latest_raw_frame = None

camera_active = False

last_camera_frame_time = 0


# ============================================================
# GLOBAL DETECTION DATA
# ============================================================

vehicles = {}

vehicle_history = []

highest_speed = 0.0

total_detected = 0


# ============================================================
# THREAD LOCK
# ============================================================

lock = threading.Lock()


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 60)
print("Loading YOLO model...")
print("=" * 60)

model = YOLO(MODEL_NAME)

print("Model:", MODEL_NAME)
print("Model loaded successfully.")
print()


# ============================================================
# LOCAL IP
# ============================================================

def get_local_ip():

    try:

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

        s.connect(
            ("8.8.8.8", 80)
        )

        ip = s.getsockname()[0]

        s.close()

        return ip

    except Exception:

        return "127.0.0.1"


# ============================================================
# DISTANCE BETWEEN POINTS
# ============================================================

def point_distance(p1, p2):

    return math.sqrt(

        (p1[0] - p2[0]) ** 2
        +
        (p1[1] - p2[1]) ** 2

    )


# ============================================================
# SPEED COLOR
# ============================================================

def get_speed_color(speed, moving=True):

    if not moving:

        return (160, 160, 160)

    if speed < NORMAL_SPEED:

        return (0, 255, 0)

    elif speed < MODERATE_SPEED:

        return (0, 255, 255)

    else:

        return (0, 0, 255)


# ============================================================
# SPEED STATUS
# ============================================================

def get_speed_status(speed, moving):

    if not moving:

        return "PARKED"

    if speed < NORMAL_SPEED:

        return "NORMAL"

    elif speed < MODERATE_SPEED:

        return "MODERATE"

    else:

        return "SPEEDING"


# ============================================================
# LINE POSITION
# ============================================================

def line_crossed(previous_y, current_y, line_y):

    if previous_y is None:

        return False

    # Vehicle moving down
    if previous_y < line_y <= current_y:

        return True

    # Vehicle moving up
    if previous_y > line_y >= current_y:

        return True

    return False


# ============================================================
# SPEED USING TWO CALIBRATION LINES
# ============================================================

def calculate_line_speed(vehicle):

    line_a_time = vehicle.get(
        "line_a_time"
    )

    line_b_time = vehicle.get(
        "line_b_time"
    )

    if (
        line_a_time is None
        or
        line_b_time is None
    ):

        return 0.0

    time_difference = abs(
        line_b_time -
        line_a_time
    )

    if time_difference <= 0:

        return 0.0

    speed_ms = (
        CALIBRATION_DISTANCE_METERS
        /
        time_difference
    )

    speed_kmh = speed_ms * 3.6

    if speed_kmh > MAX_SPEED:

        return 0.0

    return speed_kmh


# ============================================================
# DETECTION
# ============================================================

def detect_vehicles(frame):

    global highest_speed
    global total_detected

    current_time = time.time()

    try:

        results = model.track(

            source=frame,

            persist=True,

            tracker="bytetrack.yaml",

            classes=[
                1,
                2,
                3,
                5,
                7
            ],

            conf=0.15,

            iou=0.45,

            imgsz=640,

            max_det=40,

            verbose=False

        )

    except Exception as error:

        print(
            "YOLO ERROR:",
            error
        )

        return


    if not results:

        return


    result = results[0]

    boxes = result.boxes

    current_ids = set()


    if boxes is None:

        return


    for box in boxes:

        try:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

        except Exception:

            continue


        if class_id not in VEHICLE_CLASSES:

            continue


        minimum_confidence = CLASS_CONFIDENCE.get(
            class_id,
            0.20
        )


        if confidence < minimum_confidence:

            continue


        if box.id is None:

            continue


        try:

            track_id = int(
                box.id[0].item()
            )

        except Exception:

            continue


        try:

            x1, y1, x2, y2 = map(

                int,

                box.xyxy[0].tolist()

            )

        except Exception:

            continue


        width = x2 - x1
        height = y2 - y1


        if width <= 0 or height <= 0:

            continue


        # Ignore extremely large false detections
        frame_area = (
            frame.shape[0]
            *
            frame.shape[1]
        )

        box_area = width * height

        area_ratio = (
            box_area /
            frame_area
        )


        if area_ratio > 0.65:

            continue


        # ====================================================
        # VEHICLE GROUND POINT
        # ====================================================

        # Bottom center is better than box center
        # for road-speed estimation.

        ground_point = (

            int(
                (x1 + x2) / 2
            ),

            int(y2)

        )


        vehicle_type = VEHICLE_CLASSES[
            class_id
        ]


        current_ids.add(
            track_id
        )


        # ====================================================
        # NEW VEHICLE
        # ====================================================

        if track_id not in vehicles:

            vehicles[track_id] = {

                "id":
                    track_id,

                "type":
                    vehicle_type,

                "class_id":
                    class_id,

                "confidence":
                    confidence,

                "box":
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    ),

                "position":
                    ground_point,

                "previous_position":
                    None,

                "history":
                    [],

                "speed":
                    0.0,

                "max_speed":
                    0.0,

                "moving":
                    False,

                "line_a_time":
                    None,

                "line_b_time":
                    None,

                "speed_locked":
                    False,

                "last_seen":
                    current_time

            }

            total_detected += 1


        vehicle = vehicles[
            track_id
        ]


        # ====================================================
        # UPDATE
        # ====================================================

        previous_position = vehicle[
            "position"
        ]


        vehicle[
            "previous_position"
        ] = previous_position


        vehicle[
            "position"
        ] = ground_point


        vehicle[
            "box"
        ] = (
            x1,
            y1,
            x2,
            y2
        )


        vehicle[
            "confidence"
        ] = confidence


        vehicle[
            "type"
        ] = vehicle_type


        vehicle[
            "last_seen"
        ] = current_time


        vehicle[
            "history"
        ].append(

            (
                ground_point,
                current_time
            )

        )


        if len(
            vehicle["history"]
        ) > 20:

            vehicle[
                "history"
            ] = vehicle[
                "history"
            ][-20:]


        # ====================================================
        # MOVEMENT DETECTION
        # ====================================================

        movement_pixels = 0


        if previous_position:

            movement_pixels = point_distance(

                previous_position,

                ground_point

            )


        # Small movement = camera/model jitter
        if movement_pixels >= 4:

            vehicle[
                "moving"
            ] = True

        else:

            # Don't immediately call it parked.
            # Keep moving state if it was recently moving.

            if (
                current_time -
                vehicle["last_seen"]
                < 0.4
            ):

                pass


        # ====================================================
        # CALIBRATION LINE CROSSING
        # ====================================================

        previous_y = None


        if previous_position:

            previous_y = previous_position[1]


        current_y = ground_point[1]


        # Line A
        if (
            vehicle["line_a_time"]
            is None
            and
            line_crossed(
                previous_y,
                current_y,
                get_line_a_y(frame)
            )
        ):

            vehicle[
                "line_a_time"
            ] = current_time


        # Line B
        if (
            vehicle["line_a_time"]
            is not None
            and
            vehicle["line_b_time"]
            is None
            and
            line_crossed(
                previous_y,
                current_y,
                get_line_b_y(frame)
            )
        ):

            vehicle[
                "line_b_time"
            ] = current_time


        # ====================================================
        # CALCULATE SPEED
        # ====================================================

        if (
            vehicle["line_a_time"]
            is not None
            and
            vehicle["line_b_time"]
            is not None
            and
            not vehicle["speed_locked"]
        ):

            calculated_speed = calculate_line_speed(
                vehicle
            )


            if calculated_speed > 0:

                vehicle[
                    "speed"
                ] = calculated_speed


                vehicle[
                    "max_speed"
                ] = calculated_speed


                vehicle[
                    "speed_locked"
                ] = True


                if calculated_speed > highest_speed:

                    highest_speed = \
                        calculated_speed


        # ====================================================
        # IF VEHICLE HAS NOT CROSSED BOTH LINES
        # ====================================================

        # Show 0 until proper calibration measurement
        # is available.

        if not vehicle["speed_locked"]:

            if movement_pixels < 4:

                vehicle[
                    "speed"
                ] = 0.0


# ============================================================
# LINE POSITIONS
# ============================================================

def get_line_a_y(frame):

    return int(
        frame.shape[0] * 0.40
    )


def get_line_b_y(frame):

    return int(
        frame.shape[0] * 0.65
    )


# ============================================================
# REMOVE OLD VEHICLES
# ============================================================

def cleanup_vehicles():

    current_time = time.time()

    remove_ids = []


    for vehicle_id, vehicle in vehicles.items():

        if (
            current_time -
            vehicle["last_seen"]
            > 2.5
        ):

            remove_ids.append(
                vehicle_id
            )


    for vehicle_id in remove_ids:

        vehicle = vehicles[
            vehicle_id
        ]


        # Save only vehicles which
        # actually received speed data.

        if vehicle[
            "speed_locked"
        ]:

            vehicle_history.append({

                "id":
                    vehicle["id"],

                "type":
                    vehicle["type"],

                "speed":
                    round(
                        vehicle["speed"],
                        1
                    ),

                "status":
                    get_speed_status(
                        vehicle["speed"],
                        vehicle["moving"]
                    ),

                "time":
                    time.strftime(
                        "%H:%M:%S"
                    )

            })


        if len(
            vehicle_history
        ) > 100:

            del vehicle_history[:-100]


        del vehicles[
            vehicle_id
        ]


# ============================================================
# DRAW EVERYTHING
# ============================================================

def draw_overlay(frame):

    output = frame.copy()


    # ========================================================
    # CALIBRATION LINES
    # ========================================================

    line_a_y = get_line_a_y(
        output
    )

    line_b_y = get_line_b_y(
        output
    )


    cv2.line(

        output,

        (
            0,
            line_a_y
        ),

        (
            output.shape[1],
            line_a_y
        ),

        (255, 180, 0),

        2

    )


    cv2.line(

        output,

        (
            0,
            line_b_y
        ),

        (
            output.shape[1],
            line_b_y
        ),

        (255, 180, 0),

        2

    )


    cv2.putText(

        output,

        "A",

        (
            10,
            line_a_y - 8
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 180, 0),

        2

    )


    cv2.putText(

        output,

        "B",

        (
            10,
            line_b_y - 8
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 180, 0),

        2

    )


    # ========================================================
    # VEHICLES
    # ========================================================

    with lock:

        for vehicle in vehicles.values():

            x1, y1, x2, y2 = vehicle[
                "box"
            ]


            speed = vehicle[
                "speed"
            ]


            moving = vehicle[
                "moving"
            ]


            color = get_speed_color(

                speed,

                moving

            )


            status = get_speed_status(

                speed,

                moving

            )


            # ------------------------------------------------
            # BOX
            # ------------------------------------------------

            cv2.rectangle(

                output,

                (
                    x1,
                    y1
                ),

                (
                    x2,
                    y2
                ),

                color,

                2

            )


            # ------------------------------------------------
            # VEHICLE ID + TYPE
            # ------------------------------------------------

            label = (

                f'#{vehicle["id"]} '
                f'{vehicle["type"]}'

            )


            label_y = max(
                y1 - 8,
                18
            )


            cv2.putText(

                output,

                label,

                (
                    x1,
                    label_y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                color,

                2

            )


            # ------------------------------------------------
            # SPEED
            # ------------------------------------------------

            speed_text = (

                f'{int(round(speed))} km/h'

            )


            speed_y = min(

                y2 + 22,

                output.shape[0] - 30

            )


            cv2.putText(

                output,

                speed_text,

                (
                    x1,
                    speed_y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                color,

                2

            )


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            status_y = min(

                y2 + 42,

                output.shape[0] - 8

            )


            cv2.putText(

                output,

                status,

                (
                    x1,
                    status_y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.42,

                color,

                2

            )


            # ------------------------------------------------
            # GROUND POINT
            # ------------------------------------------------

            px, py = vehicle[
                "position"
            ]


            cv2.circle(

                output,

                (
                    px,
                    py
                ),

                4,

                color,

                -1

            )


    # ========================================================
    # TOP DASHBOARD
    # ========================================================

    overlay = output.copy()


    cv2.rectangle(

        overlay,

        (
            10,
            10
        ),

        (
            365,
            128
        ),

        (
            0,
            0,
            0
        ),

        -1

    )


    cv2.addWeighted(

        overlay,

        0.78,

        output,

        0.22,

        0,

        output

    )


    cv2.putText(

        output,

        "VEHICLE SPEED TRACKER",

        (
            20,
            34
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2

    )


    cv2.putText(

        output,

        f"Live Vehicles: {len(vehicles)}",

        (
            20,
            61
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (255, 255, 255),

        2

    )


    cv2.putText(

        output,

        (
            f"Highest Speed: "
            f"{int(round(highest_speed))} km/h"
        ),

        (
            20,
            87
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (255, 255, 255),

        2

    )


    cv2.putText(

        output,

        f"Total Detected: {total_detected}",

        (
            20,
            112
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (0, 255, 0),

        2

    )


    # Calibration distance
    cv2.putText(

        output,

        (
            f"Calibration: "
            f"{CALIBRATION_DISTANCE_METERS:g} m"
        ),

        (
            output.shape[1] - 170,
            25
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.42,

        (255, 180, 0),

        1

    )


    return output


# ============================================================
# DETECTION THREAD
# ============================================================

def detection_loop():

    last_detection = 0


    while True:

        with lock:

            if latest_raw_frame is None:

                frame = None

            else:

                frame = latest_raw_frame.copy()


        if frame is None:

            time.sleep(
                0.02
            )

            continue


        current_time = time.time()


        # YOLO processing interval

        if (
            current_time -
            last_detection
            >= DETECTION_INTERVAL
        ):

            last_detection = \
                current_time


            detect_vehicles(
                frame
            )


            cleanup_vehicles()


        time.sleep(
            0.005
        )


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_frames():

    blank = np.zeros(

        (
            FRAME_HEIGHT,
            FRAME_WIDTH,
            3
        ),

        dtype=np.uint8

    )


    cv2.putText(

        blank,

        "WAITING FOR PHONE CAMERA...",

        (
            100,
            170
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    cv2.putText(

        blank,

        "Open /phone on your phone",

        (
            170,
            205
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.42,

        (180, 180, 180),

        1

    )


    success, encoded = cv2.imencode(

        ".jpg",

        blank,

        [
            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),
            JPEG_QUALITY
        ]

    )


    if success:

        blank_bytes = \
            encoded.tobytes()

    else:

        blank_bytes = b""


    while True:

        with lock:

            if latest_raw_frame is not None:

                frame = \
                    latest_raw_frame.copy()

            else:

                frame = None


        if frame is None:

            frame_bytes = blank_bytes


        else:

            output = draw_overlay(
                frame
            )


            success, encoded = cv2.imencode(

                ".jpg",

                output,

                [
                    int(
                        cv2.IMWRITE_JPEG_QUALITY
                    ),
                    JPEG_QUALITY
                ]

            )


            if success:

                frame_bytes = \
                    encoded.tobytes()

            else:

                frame_bytes = \
                    blank_bytes


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n"

            b"Cache-Control: no-cache\r\n"

            b"Pragma: no-cache\r\n\r\n"

            +
            frame_bytes
            +
            b"\r\n"

        )


        time.sleep(
            1 / OUTPUT_FPS
        )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ============================================================
# PHONE
# ============================================================

@app.route("/phone")
def phone():

    return render_template(
        "phone.html"
    )


# ============================================================
# MONITOR
# ============================================================

@app.route("/monitor")
def monitor():

    return render_template(
        "index.html"
    )


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(

        generate_frames(),

        mimetype=
        "multipart/x-mixed-replace; boundary=frame",

        headers={

            "Cache-Control":
                "no-cache",

            "Pragma":
                "no-cache"

        }

    )


# ============================================================
# PHONE UPLOAD
# ============================================================

@app.route(

    "/upload_frame",

    methods=["POST"]

)
def upload_frame():

    global latest_raw_frame

    global camera_active

    global last_camera_frame_time


    try:

        if "image" not in request.files:

            return jsonify({

                "status":
                    "error",

                "message":
                    "No image"

            }), 400


        image_data = request.files[
            "image"
        ].read()


        npimg = np.frombuffer(

            image_data,

            np.uint8

        )


        frame = cv2.imdecode(

            npimg,

            cv2.IMREAD_COLOR

        )


        if frame is None:

            return jsonify({

                "status":
                    "error"

            }), 400


        # ====================================================
        # PRESERVE 16:9
        # ====================================================

        frame = cv2.resize(

            frame,

            (
                FRAME_WIDTH,
                FRAME_HEIGHT
            ),

            interpolation=cv2.INTER_AREA

        )


        with lock:

            latest_raw_frame = \
                frame

            camera_active = True

            last_camera_frame_time = \
                time.time()


        return jsonify({

            "status":
                "success"

        })


    except Exception as error:

        print(
            "UPLOAD ERROR:",
            error
        )


        return jsonify({

            "status":
                "error"

        }), 500


# ============================================================
# STATUS API
# ============================================================

@app.route("/status")
def status():

    global camera_active


    if (

        time.time()
        -
        last_camera_frame_time
        >
        3

    ):

        camera_active = False


    with lock:

        live_vehicles = []


        for vehicle in vehicles.values():

            live_vehicles.append({

                "id":
                    vehicle["id"],

                "type":
                    vehicle["type"],

                "speed":
                    round(
                        vehicle["speed"],
                        1
                    ),

                "max_speed":
                    round(
                        vehicle["max_speed"],
                        1
                    ),

                "status":
                    get_speed_status(

                        vehicle["speed"],

                        vehicle["moving"]

                    )

            })


        history = \
            vehicle_history[-30:]


    moving_speeds = [

        v["speed"]

        for v in vehicles.values()

        if v["moving"]
        and
        v["speed"] > 0

    ]


    if moving_speeds:

        average_speed = sum(
            moving_speeds
        ) / len(
            moving_speeds
        )

    else:

        average_speed = 0


    return jsonify({

        "camera_active":
            camera_active,

        "vehicles":
            live_vehicles,

        "vehicle_count":
            len(live_vehicles),

        "highest_speed":
            round(
                highest_speed,
                1
            ),

        "average_speed":
            round(
                average_speed,
                1
            ),

        "total_detected":
            total_detected,

        "history":
            history,

        "calibration_distance":
            CALIBRATION_DISTANCE_METERS

    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    local_ip = get_local_ip()


    print()

    print(
        "=" * 60
    )

    print(
        "       🚗 VEHICLE SPEED TRACKER"
    )

    print(
        "=" * 60
    )

    print()

    print(
        "📱 PHONE CAMERA:"
    )

    print(
        f"http://{local_ip}:5000/phone"
    )

    print()

    print(
        "💻 LAPTOP MONITOR:"
    )

    print(
        f"http://{local_ip}:5000/monitor"
    )

    print()

    print(
        "📊 STATUS:"
    )

    print(
        f"http://{local_ip}:5000/status"
    )

    print()

    print(
        "📏 CALIBRATION DISTANCE:"
    )

    print(
        f"{CALIBRATION_DISTANCE_METERS} meters"
    )

    print()

    print(
        "Landscape camera recommended."
    )

    print(
        "Same Wi-Fi recommended."
    )

    print()

    print(
        "=" * 60
    )

    print()


    detection_thread = threading.Thread(

        target=detection_loop,

        daemon=True

    )


    detection_thread.start()


    app.run(

        host=HOST,

        port=PORT,

        threaded=True,

        debug=False

    )
