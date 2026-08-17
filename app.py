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
# CONFIGURATION
# ============================================================

HOST = "0.0.0.0"
PORT = 5000

# Better than yolo11n for small/distant vehicles
MODEL_NAME = "yolo11s.pt"

# ------------------------------------------------------------
# SPEED CALIBRATION
# IMPORTANT:
# This is an initial estimate.
# We will calibrate this after road testing.
# ------------------------------------------------------------

PIXEL_TO_METER_RATIO = 0.05

# Speed limits for display
NORMAL_SPEED = 40
MODERATE_SPEED = 60

# Maximum allowed calculated speed
MAX_SPEED = 200

# General YOLO confidence
CONFIDENCE = 0.15

# ------------------------------------------------------------
# Vehicle classes from COCO
# ------------------------------------------------------------

VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Different confidence for different vehicles
# Motorcycle gets lower threshold because
# bikes are often smaller/farther from camera.

CLASS_CONFIDENCE = {
    1: 0.18,   # Bicycle
    2: 0.25,   # Car
    3: 0.15,   # Motorcycle
    5: 0.30,   # Bus
    7: 0.30    # Truck
}


# ============================================================
# YOLO MODEL
# ============================================================

print()
print("=" * 60)
print("Loading YOLO model...")
print("=" * 60)

try:
    model = YOLO(MODEL_NAME)
    print("YOLO model loaded successfully.")
    print("Model:", MODEL_NAME)

except Exception as e:
    print("ERROR loading YOLO model:")
    print(e)
    raise


# ============================================================
# GLOBAL VARIABLES
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
# GET LOCAL IP
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
# SPEED COLOR
# ============================================================

def get_speed_color(speed):

    if speed < NORMAL_SPEED:

        # Green
        return (0, 255, 0)

    elif speed < MODERATE_SPEED:

        # Yellow
        return (0, 255, 255)

    else:

        # Red
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
# PIXEL DISTANCE
# ============================================================

def pixel_distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# SPEED CALCULATION
# ============================================================

def calculate_vehicle_speed(vehicle):

    history = vehicle["history"]

    # Need enough history
    if len(history) < 3:

        return 0.0

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

    # Ignore tiny detection jitter
    if distance_pixels < 3:

        return 0.0

    # Convert pixels -> meters
    distance_meters = (
        distance_pixels *
        PIXEL_TO_METER_RATIO
    )

    # m/s
    speed_ms = (
        distance_meters /
        time_difference
    )

    # km/h
    speed_kmh = (
        speed_ms *
        3.6
    )

    # Ignore impossible spikes
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
    # YOLO TRACKING
    # ========================================================

    results = model.track(

        source=frame,

        persist=True,

        tracker="bytetrack.yaml",

        classes=[
            1,  # Bicycle
            2,  # Car
            3,  # Motorcycle
            5,  # Bus
            7   # Truck
        ],

        conf=CONFIDENCE,

        iou=0.45,

        # Larger image helps small vehicles
        imgsz=960,

        # Allow multiple vehicles
        max_det=50,

        verbose=False
    )

    if not results:

        return frame

    result = results[0]

    boxes = result.boxes

    # ========================================================
    # PROCESS DETECTED VEHICLES
    # ========================================================

    if boxes is not None and len(boxes) > 0:

        for box in boxes:

            # ------------------------------------------------
            # CLASS ID
            # ------------------------------------------------

            try:

                class_id = int(
                    box.cls[0].item()
                )

            except Exception:

                continue

            if class_id not in VEHICLE_CLASSES:

                continue

            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            try:

                confidence = float(
                    box.conf[0].item()
                )

            except Exception:

                continue

            required_confidence = \
                CLASS_CONFIDENCE.get(
                    class_id,
                    0.25
                )

            if confidence < required_confidence:

                continue

            vehicle_type = \
                VEHICLE_CLASSES[class_id]

            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            try:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

            except Exception:

                continue

            box_width = x2 - x1

            box_height = y2 - y1

            if box_width <= 0 or box_height <= 0:

                continue

            # ------------------------------------------------
            # REMOVE VERY LARGE SUSPICIOUS BOXES
            # ------------------------------------------------

            box_area = (
                box_width *
                box_height
            )

            frame_area = (
                frame.shape[0] *
                frame.shape[1]
            )

            area_ratio = (
                box_area /
                frame_area
            )

            # A giant detection can often be
            # a false detection.
            if area_ratio > 0.55:

                continue

            # ------------------------------------------------
            # CENTER
            # ------------------------------------------------

            center = (
                int((x1 + x2) / 2),
                int((y1 + y2) / 2)
            )

            # ------------------------------------------------
            # TRACKING ID
            # ------------------------------------------------

            track_id = None

            if box.id is not None:

                try:

                    track_id = int(
                        box.id[0].item()
                    )

                except Exception:

                    track_id = None

            # =================================================
            # IF TRACK ID NOT AVAILABLE
            # =================================================

            if track_id is None:

                # Still draw the detection.
                # Speed cannot be calculated
                # without a tracking ID.

                color = (
                    255,
                    255,
                    255
                )

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
                    (
                        x1,
                        max(y1 - 8, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

                continue

            # =================================================
            # CREATE / UPDATE VEHICLE
            # =================================================

            with lock:

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

                        "last_seen":
                            current_time,

                        "movement":
                            False,

                        "confidence":
                            confidence
                    }

                    total_detected += 1

                vehicle = vehicles[
                    track_id
                ]

                # ------------------------------------------------
                # UPDATE DATA
                # ------------------------------------------------

                vehicle["position"] = center

                vehicle["type"] = vehicle_type

                vehicle["confidence"] = confidence

                vehicle["last_seen"] = current_time

                vehicle["history"].append(
                    (
                        center,
                        current_time
                    )
                )

                # Keep only recent positions
                if len(
                    vehicle["history"]
                ) > 12:

                    vehicle["history"] = \
                        vehicle["history"][-12:]

                # =================================================
                # MOVEMENT DETECTION
                # =================================================

                movement_distance = 0

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

                # Small movement can simply be
                # detection jitter.
                if movement_distance >= 5:

                    vehicle["movement"] = True

                else:

                    vehicle["movement"] = False

                # =================================================
                # SPEED
                # =================================================

                calculated_speed = \
                    calculate_vehicle_speed(
                        vehicle
                    )

                # If parked / not moving
                # force speed to zero.
                if not vehicle["movement"]:

                    calculated_speed = 0.0

                # =================================================
                # SPEED SMOOTHING
                # =================================================

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

                # Remove tiny fake speeds
                if calculated_speed < 2:

                    calculated_speed = 0.0

                vehicle["speed"] = \
                    calculated_speed

                # =================================================
                # MAX SPEED
                # =================================================

                if calculated_speed > \
                        vehicle["max_speed"]:

                    vehicle["max_speed"] = \
                        calculated_speed

                # =================================================
                # GLOBAL HIGHEST SPEED
                # =================================================

                if calculated_speed > \
                        highest_speed:

                    highest_speed = \
                        calculated_speed

            # =================================================
            # DRAW VEHICLE BOX
            # =================================================

            speed = vehicle["speed"]

            moving = vehicle["movement"]

            color = get_speed_color(
                speed
            )

            status = get_speed_status(
                speed,
                moving
            )

            # ------------------------------------------------
            # BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # ------------------------------------------------
            # VEHICLE TYPE + ID
            # ------------------------------------------------

            top_label = (
                f"#{track_id} "
                f"{vehicle_type}"
            )

            cv2.putText(
                frame,
                top_label,
                (
                    x1,
                    max(
                        y1 - 10,
                        22
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                color,
                2
            )

            # ------------------------------------------------
            # SPEED
            # ------------------------------------------------

            speed_label = (
                f"{int(round(speed))} km/h"
            )

            speed_y = min(
                y2 + 25,
                frame.shape[0] - 35
            )

            cv2.putText(
                frame,
                speed_label,
                (
                    x1,
                    speed_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            status_y = min(
                y2 + 48,
                frame.shape[0] - 10
            )

            cv2.putText(
                frame,
                status,
                (
                    x1,
                    status_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2
            )

    # ========================================================
    # REMOVE OLD VEHICLES
    # ========================================================

    with lock:

        expired_ids = []

        for vid, vehicle in vehicles.items():

            if (
                current_time -
                vehicle["last_seen"]
                > 2.5
            ):

                expired_ids.append(vid)

        for vid in expired_ids:

            vehicle = vehicles[vid]

            final_speed = max(
                vehicle["speed"],
                vehicle["max_speed"]
            )

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
                        final_speed,
                        final_speed > 2
                    ),

                "time":
                    time.strftime(
                        "%H:%M:%S"
                    )
            })

            # Keep last 100 records
            if len(vehicle_history) > 100:

                del vehicle_history[:-100]

            del vehicles[vid]

    # ========================================================
    # TOP INFORMATION PANEL
    # ========================================================

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (360, 125),
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

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "VEHICLE SPEED TRACKER",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # LIVE VEHICLES
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Live Vehicles: {len(vehicles)}",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # HIGHEST SPEED
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Highest Speed: "
        f"{int(highest_speed)} km/h",
        (20, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # TOTAL DETECTED
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Total Detected: "
        f"{total_detected}",
        (20, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    return frame


# ============================================================
# FRAME PROCESSING THREAD
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

            processed_frame = \
                process_frame(frame)

            success, encoded = \
                cv2.imencode(
                    ".jpg",
                    processed_frame,
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

            print()
            print(
                "PROCESSING ERROR:",
                e
            )
            print()


processor_thread = threading.Thread(
    target=frame_processor,
    daemon=True
)

processor_thread.start()


# ============================================================
# VIDEO GENERATOR
# ============================================================

def generate_frames():

    # Blank frame before phone connects

    blank_frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    cv2.putText(
        blank_frame,
        "WAITING FOR PHONE CAMERA...",
        (90, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        blank_frame,
        "Open /phone on your phone",
        (145, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (170, 170, 170),
        1
    )

    _, encoded = cv2.imencode(
        ".jpg",
        blank_frame
    )

    blank_bytes = encoded.tobytes()

    while True:

        with lock:

            frame_bytes = \
                latest_processed_frame

        if frame_bytes is None:

            frame_bytes = blank_bytes

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            +
            frame_bytes
            +
            b"\r\n"
        )

        time.sleep(0.03)


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
        "multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# RECEIVE FRAME FROM PHONE
# ============================================================

@app.route(
    "/upload_frame",
    methods=["POST"]
)
def upload_frame():

    global camera_active
    global last_frame_time

    try:

        # ----------------------------------------------------
        # Check image
        # ----------------------------------------------------

        if "image" not in request.files:

            return jsonify({

                "status":
                    "error",

                "message":
                    "No image received"

            }), 400

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

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
                    "error",

                "message":
                    "Invalid image"

            }), 400

        # ----------------------------------------------------
        # Resize to processing size
        # ----------------------------------------------------

        frame = cv2.resize(
            frame,
            (640, 480),
            interpolation=cv2.INTER_AREA
        )

        # ----------------------------------------------------
        # Keep queue small
        # ----------------------------------------------------

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

            "status":
                "success"

        })

    except Exception as e:

        print(
            "UPLOAD ERROR:",
            e
        )

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# ============================================================
# STATUS API
# ============================================================

@app.route("/status")
def status():

    global camera_active

    # Camera timeout
    if (
        time.time()
        -
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
# START SERVER
# ============================================================

if __name__ == "__main__":

    local_ip = get_local_ip()

    print()
    print("=" * 60)
    print("          🚗 VEHICLE SPEED TRACKER V3")
    print("=" * 60)

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
        "📊 STATUS API:"
    )

    print(
        f"http://{local_ip}:5000/status"
    )

    print()

    print(
        "Both phone and laptop must be"
    )

    print(
        "connected to the same Wi-Fi."
    )

    print()

    print("=" * 60)
    print()

    app.run(
        host=HOST,
        port=PORT,
        threaded=True
    )
