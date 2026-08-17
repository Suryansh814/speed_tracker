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
# CONFIG
# ============================================================

HOST = "0.0.0.0"
PORT = 5000

MODEL_NAME = "yolo11s.pt"

# Processing resolution
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# YOLO input size
YOLO_SIZE = 640

# YOLO detection interval
# YOLO does not need to process every single camera frame.
DETECTION_INTERVAL = 0.10

# JPEG quality
JPEG_QUALITY = 72

# Speed calibration
PIXEL_TO_METER_RATIO = 0.05

NORMAL_SPEED = 40
MODERATE_SPEED = 60

MAX_SPEED = 200


# ============================================================
# VEHICLE CLASSES
# ============================================================

VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


CLASS_CONFIDENCE = {
    1: 0.18,
    2: 0.25,
    3: 0.15,
    5: 0.30,
    7: 0.30
}


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("Loading YOLO model...")
print("=" * 60)

model = YOLO(MODEL_NAME)

print("Model loaded:", MODEL_NAME)
print()


# ============================================================
# GLOBAL DATA
# ============================================================

latest_raw_frame = None
latest_output_frame = None

last_detection_time = 0

camera_active = False
last_camera_frame_time = 0

vehicles = {}

vehicle_history = []

highest_speed = 0
total_detected = 0

lock = threading.Lock()


# ============================================================
# LOCAL IP
# ============================================================

def get_local_ip():

    try:

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

        s.connect(("8.8.8.8", 80))

        ip = s.getsockname()[0]

        s.close()

        return ip

    except:

        return "127.0.0.1"


# ============================================================
# DISTANCE
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# SPEED COLOR
# ============================================================

def speed_color(speed):

    if speed < NORMAL_SPEED:

        return (0, 255, 0)

    elif speed < MODERATE_SPEED:

        return (0, 255, 255)

    else:

        return (0, 0, 255)


# ============================================================
# SPEED STATUS
# ============================================================

def speed_status(speed, moving):

    if not moving:

        return "PARKED"

    if speed < NORMAL_SPEED:

        return "NORMAL"

    elif speed < MODERATE_SPEED:

        return "MODERATE"

    else:

        return "SPEEDING"


# ============================================================
# CALCULATE SPEED
# ============================================================

def calculate_speed(vehicle):

    history = vehicle["history"]

    if len(history) < 3:

        return 0.0

    current_pos, current_time = history[-1]

    old_pos, old_time = history[-3]

    time_difference = current_time - old_time

    if time_difference <= 0:

        return 0.0

    pixel_distance = distance(
        current_pos,
        old_pos
    )

    # Ignore tiny camera/detection jitter
    if pixel_distance < 4:

        return 0.0

    meters = (
        pixel_distance *
        PIXEL_TO_METER_RATIO
    )

    meters_per_second = (
        meters /
        time_difference
    )

    kmh = (
        meters_per_second *
        3.6
    )

    if kmh > MAX_SPEED:

        return vehicle["speed"]

    return kmh


# ============================================================
# DETECTION FUNCTION
# ============================================================

def detect_and_update(frame):

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

            imgsz=YOLO_SIZE,

            max_det=50,

            verbose=False
        )

    except Exception as e:

        print("YOLO ERROR:", e)

        return

    if not results:

        return

    result = results[0]

    boxes = result.boxes

    current_ids = set()

    if boxes is not None and len(boxes) > 0:

        for box in boxes:

            try:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

            except:

                continue

            if class_id not in VEHICLE_CLASSES:

                continue

            required_confidence = CLASS_CONFIDENCE.get(
                class_id,
                0.25
            )

            if confidence < required_confidence:

                continue

            try:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

            except:

                continue

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:

                continue

            # Ignore extremely huge detections
            area_ratio = (
                (width * height) /
                (frame.shape[0] * frame.shape[1])
            )

            if area_ratio > 0.55:

                continue

            # Center of vehicle
            center = (
                int((x1 + x2) / 2),
                int((y1 + y2) / 2)
            )

            # Track ID
            if box.id is None:

                continue

            try:

                track_id = int(
                    box.id[0].item()
                )

            except:

                continue

            current_ids.add(track_id)

            vehicle_type = VEHICLE_CLASSES[
                class_id
            ]

            # =================================================
            # NEW VEHICLE
            # =================================================

            if track_id not in vehicles:

                vehicles[track_id] = {

                    "id":
                        track_id,

                    "type":
                        vehicle_type,

                    "position":
                        center,

                    "history": [
                        (
                            center,
                            current_time
                        )
                    ],

                    "speed":
                        0.0,

                    "max_speed":
                        0.0,

                    "movement":
                        False,

                    "last_seen":
                        current_time,

                    "confidence":
                        confidence,

                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                }

                total_detected += 1

            # =================================================
            # EXISTING VEHICLE
            # =================================================

            vehicle = vehicles[track_id]

            vehicle["type"] = vehicle_type

            vehicle["position"] = center

            vehicle["last_seen"] = current_time

            vehicle["confidence"] = confidence

            vehicle["box"] = (
                x1,
                y1,
                x2,
                y2
            )

            vehicle["history"].append(
                (
                    center,
                    current_time
                )
            )

            if len(vehicle["history"]) > 12:

                vehicle["history"] = \
                    vehicle["history"][-12:]

            # =================================================
            # MOVEMENT
            # =================================================

            if len(vehicle["history"]) >= 3:

                old_position = vehicle[
                    "history"
                ][-3][0]

                movement = distance(
                    old_position,
                    center
                )

            else:

                movement = 0

            if movement >= 5:

                vehicle["movement"] = True

            else:

                vehicle["movement"] = False

            # =================================================
            # SPEED
            # =================================================

            calculated_speed = calculate_speed(
                vehicle
            )

            # Parked = zero
            if not vehicle["movement"]:

                calculated_speed = 0

            # Remove tiny fake speed
            if calculated_speed < 2:

                calculated_speed = 0

            # Smooth speed
            old_speed = vehicle["speed"]

            if (
                vehicle["movement"]
                and old_speed > 0
            ):

                calculated_speed = (
                    old_speed * 0.65
                    +
                    calculated_speed * 0.35
                )

            vehicle["speed"] = \
                calculated_speed

            # Maximum speed
            if calculated_speed > \
                    vehicle["max_speed"]:

                vehicle["max_speed"] = \
                    calculated_speed

            if calculated_speed > \
                    highest_speed:

                highest_speed = \
                    calculated_speed

    # =========================================================
    # REMOVE VEHICLES THAT DISAPPEARED
    # =========================================================

    expired = []

    for vid, vehicle in vehicles.items():

        if (
            current_time -
            vehicle["last_seen"]
            > 2.0
        ):

            expired.append(vid)

    for vid in expired:

        vehicle = vehicles[vid]

        vehicle_history.append({

            "id":
                vehicle["id"],

            "type":
                vehicle["type"],

            "speed":
                round(
                    vehicle["max_speed"],
                    1
                ),

            "time":
                time.strftime(
                    "%H:%M:%S"
                )
        })

        if len(vehicle_history) > 100:

            del vehicle_history[:-100]

        del vehicles[vid]


# ============================================================
# DRAW OVERLAY
# ============================================================

def draw_overlay(frame):

    output = frame.copy()

    # ========================================================
    # VEHICLE BOXES
    # ========================================================

    with lock:

        for vehicle in vehicles.values():

            x1, y1, x2, y2 = vehicle["box"]

            speed = vehicle["speed"]

            color = speed_color(speed)

            status = speed_status(
                speed,
                vehicle["movement"]
            )

            # Box
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # Vehicle label
            label = (
                f'#{vehicle["id"]} '
                f'{vehicle["type"]}'
            )

            cv2.putText(
                output,
                label,
                (
                    x1,
                    max(y1 - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

            # Speed
            speed_text = (
                f'{int(round(speed))} km/h'
            )

            speed_y = min(
                y2 + 24,
                output.shape[0] - 35
            )

            cv2.putText(
                output,
                speed_text,
                (
                    x1,
                    speed_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                color,
                2
            )

            # Status
            status_y = min(
                y2 + 45,
                output.shape[0] - 10
            )

            cv2.putText(
                output,
                status,
                (
                    x1,
                    status_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2
            )

    # ========================================================
    # TOP PANEL
    # ========================================================

    overlay = output.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (350, 120),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.72,
        output,
        0.28,
        0,
        output
    )

    cv2.putText(
        output,
        "VEHICLE SPEED TRACKER",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Live Vehicles: {len(vehicles)}",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Highest Speed: {int(highest_speed)} km/h",
        (20, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Total Detected: {total_detected}",
        (20, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 0),
        2
    )

    return output


# ============================================================
# DETECTION THREAD
# ============================================================

def detection_loop():

    global latest_output_frame
    global last_detection_time

    while True:

        with lock:

            if latest_raw_frame is None:

                frame = None

            else:

                frame = latest_raw_frame.copy()

        if frame is None:

            time.sleep(0.02)

            continue

        current_time = time.time()

        if (
            current_time -
            last_detection_time
            >= DETECTION_INTERVAL
        ):

            last_detection_time = current_time

            detect_and_update(frame)

        time.sleep(0.005)


# ============================================================
# OUTPUT STREAM
# ============================================================

def generate_frames():

    global latest_output_frame

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
        (90, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2
    )

    cv2.putText(
        blank,
        "Open /phone on your phone",
        (145, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (180, 180, 180),
        1
    )

    _, blank_encoded = cv2.imencode(
        ".jpg",
        blank,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            JPEG_QUALITY
        ]
    )

    blank_bytes = blank_encoded.tobytes()

    while True:

        with lock:

            if latest_raw_frame is not None:

                frame = latest_raw_frame.copy()

            else:

                frame = None

        if frame is None:

            frame_bytes = blank_bytes

        else:

            # Draw latest detection data
            output = draw_overlay(frame)

            success, encoded = cv2.imencode(
                ".jpg",
                output,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    JPEG_QUALITY
                ]
            )

            if success:

                frame_bytes = encoded.tobytes()

            else:

                frame_bytes = blank_bytes

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n\r\n"
            +
            frame_bytes
            +
            b"\r\n"
        )

        # ~20 FPS output
        time.sleep(0.05)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template("home.html")


# ============================================================
# PHONE
# ============================================================

@app.route("/phone")
def phone():

    return render_template("phone.html")


# ============================================================
# MONITOR
# ============================================================

@app.route("/monitor")
def monitor():

    return render_template("index.html")


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
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }
    )


# ============================================================
# PHONE FRAME UPLOAD
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
                "status": "error"
            }), 400

        data = request.files[
            "image"
        ].read()

        npimg = np.frombuffer(
            data,
            np.uint8
        )

        frame = cv2.imdecode(
            npimg,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({
                "status": "error"
            }), 400

        # Keep fixed processing size
        frame = cv2.resize(
            frame,
            (
                FRAME_WIDTH,
                FRAME_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )

        with lock:

            latest_raw_frame = frame

            camera_active = True

            last_camera_frame_time = \
                time.time()

        return jsonify({
            "status": "success"
        })

    except Exception as e:

        print(
            "UPLOAD ERROR:",
            e
        )

        return jsonify({
            "status": "error"
        }), 500


# ============================================================
# STATUS
# ============================================================

@app.route("/status")
def status():

    global camera_active

    if (
        time.time() -
        last_camera_frame_time
        > 3
    ):

        camera_active = False

    with lock:

        live = []

        for vehicle in vehicles.values():

            live.append({

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
                    speed_status(
                        vehicle["speed"],
                        vehicle["movement"]
                    )
            })

        history = vehicle_history[-30:]

    return jsonify({

        "camera_active":
            camera_active,

        "vehicles":
            live,

        "vehicle_count":
            len(live),

        "highest_speed":
            round(
                highest_speed,
                1
            ),

        "total_detected":
            total_detected,

        "history":
            history
    })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    local_ip = get_local_ip()

    print()
    print("=" * 60)
    print("        🚗 VEHICLE SPEED TRACKER")
    print("=" * 60)

    print()
    print(
        "PHONE CAMERA:"
    )

    print(
        f"http://{local_ip}:5000/phone"
    )

    print()
    print(
        "LAPTOP MONITOR:"
    )

    print(
        f"http://{local_ip}:5000/monitor"
    )

    print()
    print(
        "STATUS:"
    )

    print(
        f"http://{local_ip}:5000/status"
    )

    print()
    print(
        "Same Wi-Fi recommended."
    )

    print("=" * 60)
    print()

    # Start detection in background
    thread = threading.Thread(
        target=detection_loop,
        daemon=True
    )

    thread.start()

    app.run(
        host=HOST,
        port=PORT,
        threaded=True,
        debug=False
    )
