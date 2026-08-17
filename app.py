import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import threading
import queue
import socket
import time
import math

app = Flask(__name__)

# ============================================================
# CONFIG
# ============================================================

HOST = "0.0.0.0"
PORT = 5000

MODEL_NAME = "yolo11n.pt"

# IMPORTANT:
# This is only an initial estimate.
# We will calibrate this after your road test.
PIXEL_TO_METER_RATIO = 0.05

# Speed thresholds
NORMAL_SPEED = 40
MODERATE_SPEED = 60

MAX_SPEED = 200

# Detection confidence
CONFIDENCE = 0.25

# ============================================================
# YOLO
# ============================================================

print("\nLoading YOLO model...")

model = YOLO(MODEL_NAME)

print("YOLO model loaded.\n")


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
# GLOBAL DATA
# ============================================================

vehicles = {}

vehicle_history = []

frame_queue = queue.Queue(maxsize=2)

latest_processed_frame = None

camera_active = False

last_frame_time = 0

highest_speed = 0

total_detected = 0

lock = threading.Lock()


# ============================================================
# NETWORK
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

    except Exception:

        return "127.0.0.1"


# ============================================================
# SPEED HELPERS
# ============================================================

def get_speed_color(speed):

    if speed < NORMAL_SPEED:

        return (0, 255, 0)

    elif speed < MODERATE_SPEED:

        return (0, 255, 255)

    else:

        return (0, 0, 255)


def get_speed_status(speed, movement):

    if not movement:

        return "PARKED"

    if speed < NORMAL_SPEED:

        return "NORMAL"

    elif speed < MODERATE_SPEED:

        return "MODERATE"

    else:

        return "SPEEDING"


# ============================================================
# DISTANCE
# ============================================================

def pixel_distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# SPEED CALCULATION
# ============================================================

def calculate_vehicle_speed(vehicle):

    history = vehicle["history"]

    if len(history) < 3:

        return 0.0

    # Use several frames instead of
    # only the previous frame.
    current_pos, current_time = history[-1]

    previous_pos, previous_time = history[-3]

    time_difference = (
        current_time -
        previous_time
    )

    if time_difference <= 0:

        return 0.0

    distance_pixels = pixel_distance(
        current_pos,
        previous_pos
    )

    # Very small movement is usually
    # camera/model jitter.
    if distance_pixels < 2.5:

        return 0.0

    distance_meters = (
        distance_pixels *
        PIXEL_TO_METER_RATIO
    )

    speed_ms = (
        distance_meters /
        time_difference
    )

    speed_kmh = speed_ms * 3.6

    # Ignore impossible spikes.
    if speed_kmh > MAX_SPEED:

        return vehicle["speed"]

    return speed_kmh


# ============================================================
# PROCESS FRAME
# ============================================================

def process_frame(frame):

    global highest_speed
    global total_detected

    current_time = time.time()

    # ========================================================
    # YOLO TRACK
    # ========================================================

    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=list(
            VEHICLE_CLASSES.keys()
        ),
        conf=CONFIDENCE,
        iou=0.50,
        verbose=False
    )

    if not results:

        return frame

    result = results[0]

    boxes = result.boxes

    current_ids = set()

    # ========================================================
    # DETECTIONS
    # ========================================================

    if boxes is not None and len(boxes) > 0:

        for box in boxes:

            # ------------------------------------------------
            # Get class
            # ------------------------------------------------

            class_id = int(
                box.cls[0].item()
            )

            if class_id not in VEHICLE_CLASSES:

                continue

            vehicle_type = VEHICLE_CLASSES[
                class_id
            ]

            confidence = float(
                box.conf[0].item()
            )

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            center = (
                int((x1 + x2) / 2),
                int((y1 + y2) / 2)
            )

            # ------------------------------------------------
            # Tracking ID
            # ------------------------------------------------

            if box.id is not None:

                track_id = int(
                    box.id[0].item()
                )

            else:

                # If tracker doesn't give ID,
                # don't calculate speed.
                track_id = None

            if track_id is None:

                # Still DRAW detection box.
                color = (255, 255, 255)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                label = (
                    f"{vehicle_type} "
                    f"{int(confidence * 100)}%"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

                continue

            current_ids.add(track_id)

            # =================================================
            # CREATE VEHICLE
            # =================================================

            with lock:

                if track_id not in vehicles:

                    vehicles[track_id] = {

                        "id": track_id,

                        "type": vehicle_type,

                        "position": center,

                        "history": [
                            (
                                center,
                                current_time
                            )
                        ],

                        "speed": 0.0,

                        "max_speed": 0.0,

                        "last_seen":
                            current_time,

                        "movement": False,

                        "confidence":
                            confidence
                    }

                    total_detected += 1

                vehicle = vehicles[
                    track_id
                ]

                # ------------------------------------------------
                # Update vehicle
                # ------------------------------------------------

                vehicle["position"] = center

                vehicle["type"] = vehicle_type

                vehicle["confidence"] = confidence

                vehicle["last_seen"] = \
                    current_time

                vehicle["history"].append(
                    (
                        center,
                        current_time
                    )
                )

                # Keep recent positions
                if len(
                    vehicle["history"]
                ) > 12:

                    vehicle["history"] = \
                        vehicle[
                            "history"
                        ][-12:]

                # ------------------------------------------------
                # Calculate movement
                # ------------------------------------------------

                if len(
                    vehicle["history"]
                ) >= 3:

                    old_pos = vehicle[
                        "history"
                    ][-3][0]

                    movement_distance = \
                        pixel_distance(
                            old_pos,
                            center
                        )

                    # Threshold prevents
                    # detection jitter from
                    # creating fake speed.
                    if movement_distance >= 3:

                        vehicle[
                            "movement"
                        ] = True

                    else:

                        vehicle[
                            "movement"
                        ] = False

                # ------------------------------------------------
                # Calculate speed
                # ------------------------------------------------

                calculated_speed = \
                    calculate_vehicle_speed(
                        vehicle
                    )

                # If vehicle isn't really moving,
                # force speed to zero.
                if not vehicle["movement"]:

                    calculated_speed = 0.0

                # ------------------------------------------------
                # Smooth moving speed
                # ------------------------------------------------

                old_speed = \
                    vehicle["speed"]

                if (
                    vehicle["movement"]
                    and old_speed > 0
                ):

                    calculated_speed = (
                        old_speed * 0.70
                        +
                        calculated_speed * 0.30
                    )

                # Prevent tiny values
                # from showing as 1-2 km/h.
                if calculated_speed < 2:

                    calculated_speed = 0.0

                vehicle["speed"] = \
                    calculated_speed

                # Maximum speed
                if calculated_speed > \
                        vehicle["max_speed"]:

                    vehicle["max_speed"] = \
                        calculated_speed

                # Global highest
                if calculated_speed > \
                        highest_speed:

                    highest_speed = \
                        calculated_speed

            # =================================================
            # DRAW VEHICLE
            # =================================================

            speed = vehicle["speed"]

            movement = vehicle["movement"]

            color = get_speed_color(
                speed
            )

            status = get_speed_status(
                speed,
                movement
            )

            # -------------------------------------------------
            # BOX
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # -------------------------------------------------
            # TOP LABEL
            # -------------------------------------------------

            vehicle_label = (
                f"#{track_id} "
                f"{vehicle_type}"
            )

            cv2.putText(
                frame,
                vehicle_label,
                (
                    x1,
                    max(y1 - 10, 22)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                color,
                2
            )

            # -------------------------------------------------
            # SPEED
            # -------------------------------------------------

            speed_label = (
                f"{int(round(speed))} km/h"
            )

            cv2.putText(
                frame,
                speed_label,
                (
                    x1,
                    min(
                        y2 + 25,
                        frame.shape[0] - 35
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

            # -------------------------------------------------
            # STATUS
            # -------------------------------------------------

            cv2.putText(
                frame,
                status,
                (
                    x1,
                    min(
                        y2 + 48,
                        frame.shape[0] - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2
            )

    # ========================================================
    # REMOVE VEHICLES
    # ========================================================

    with lock:

        expired_ids = []

        for vid, vehicle in vehicles.items():

            if (
                current_time -
                vehicle["last_seen"]
                > 2.0
            ):

                expired_ids.append(vid)

        for vid in expired_ids:

            vehicle = vehicles[vid]

            vehicle_history.append({

                "id":
                    vehicle["id"],

                "type":
                    vehicle["type"],

                "max_speed":
                    round(
                        vehicle["max_speed"],
                        1
                    ),

                "status":
                    get_speed_status(
                        vehicle["max_speed"],
                        vehicle["max_speed"] > 2
                    ),

                "time":
                    time.strftime(
                        "%H:%M:%S"
                    )
            })

            # Keep history manageable
            if len(vehicle_history) > 100:

                del vehicle_history[:-100]

            del vehicles[vid]

    # ========================================================
    # OVERLAY
    # ========================================================

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (350, 125),
        (10, 10, 10),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.78,
        frame,
        0.22,
        0,
        frame
    )

    cv2.putText(
        frame,
        "VEHICLE SPEED TRACKER",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Live Vehicles: {len(vehicles)}",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Highest Speed: {int(highest_speed)} km/h",
        (20, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Total Detected: {total_detected}",
        (20, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    return frame


# ============================================================
# FRAME PROCESSOR THREAD
# ============================================================

def frame_processor():

    global latest_processed_frame

    while True:

        try:

            frame = frame_queue.get(
                timeout=1
            )

        except queue.Empty:

            continue

        try:

            processed = process_frame(
                frame
            )

            success, encoded = \
                cv2.imencode(
                    ".jpg",
                    processed,
                    [
                        int(
                            cv2.IMWRITE_JPEG_QUALITY
                        ),
                        80
                    ]
                )

            if success:

                with lock:

                    latest_processed_frame = \
                        encoded.tobytes()

        except Exception as e:

            print(
                "\nPROCESSING ERROR:",
                e
            )


processor_thread = threading.Thread(
    target=frame_processor,
    daemon=True
)

processor_thread.start()


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_frames():

    blank = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    cv2.putText(
        blank,
        "WAITING FOR PHONE CAMERA...",
        (95, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        blank,
        "Open /phone on your phone",
        (145, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (170, 170, 170),
        1
    )

    _, encoded = cv2.imencode(
        ".jpg",
        blank
    )

    blank_bytes = encoded.tobytes()

    while True:

        with lock:

            frame = \
                latest_processed_frame

        if frame is None:

            frame_bytes = blank_bytes

        else:

            frame_bytes = frame

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

        time.sleep(0.03)


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


@app.route("/phone")
def phone():

    return render_template(
        "phone.html"
    )


@app.route("/monitor")
def monitor():

    return render_template(
        "index.html"
    )


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# PHONE FRAME UPLOAD
# ============================================================

@app.route(
    "/upload_frame",
    methods=["POST"]
)
def upload_frame():

    global camera_active
    global last_frame_time

    try:

        if "image" not in request.files:

            return jsonify({
                "status": "error",
                "message": "No image received"
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
                "status": "error",
                "message": "Invalid image"
            }), 400

        # Keep processing resolution fixed
        frame = cv2.resize(
            frame,
            (640, 480),
            interpolation=cv2.INTER_AREA
        )

        # Drop oldest frame if queue is full
        if frame_queue.full():

            try:

                frame_queue.get_nowait()

            except queue.Empty:

                pass

        try:

            frame_queue.put_nowait(
                frame
            )

        except queue.Full:

            pass

        camera_active = True

        last_frame_time = time.time()

        return jsonify({
            "status": "success"
        })

    except Exception as e:

        print(
            "UPLOAD ERROR:",
            e
        )

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ============================================================
# STATUS API
# ============================================================

@app.route("/status")
def status():

    global camera_active

    if (
        time.time() -
        last_frame_time
        > 3
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
                        vehicle["movement"]
                    )
            })

        history = \
            vehicle_history[-30:]

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

    print("\n")
    print("=" * 60)
    print("       VEHICLE SPEED TRACKER V3")
    print("=" * 60)

    print(
        f"\n📱 PHONE CAMERA:"
        f"\nhttp://{local_ip}:5000/phone"
    )

    print(
        f"\n💻 LAPTOP MONITOR:"
        f"\nhttp://{local_ip}:5000/monitor"
    )

    print(
        "\nBoth devices must be connected"
    )

    print(
        "to the same Wi-Fi network."
    )

    print("=" * 60)
    print()

    app.run(
        host=HOST,
        port=PORT,
        threaded=True
    )
