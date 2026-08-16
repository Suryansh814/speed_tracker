import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import threading
import queue
import socket
import time

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

HOST = "0.0.0.0"
PORT = 5000

# YOLO model
MODEL_NAME = "yolo11n.pt"

# ------------------------------------------------------------
# SPEED CALIBRATION
# ------------------------------------------------------------
# IMPORTANT:
# This value depends on your camera position and road.
# We will calibrate this properly after road testing.
#
# Example:
# 0.05 means approximately 5 cm per pixel.
#
PIXEL_TO_METER_RATIO = 0.05

# Speed limits
NORMAL_SPEED = 40
MODERATE_SPEED = 60

# Maximum speed accepted
MAX_SPEED = 200


# ============================================================
# YOLO
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_NAME)

print("YOLO model loaded.")


# ============================================================
# COCO VEHICLE CLASSES
# ============================================================

VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# ============================================================
# GLOBAL STATE
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
# SPEED COLOR
# ============================================================

def get_speed_color(speed):

    if speed < NORMAL_SPEED:

        return (0, 255, 0)

    elif speed < MODERATE_SPEED:

        return (0, 255, 255)

    else:

        return (0, 0, 255)


def get_speed_status(speed):

    if speed < NORMAL_SPEED:

        return "NORMAL"

    elif speed < MODERATE_SPEED:

        return "MODERATE"

    else:

        return "SPEEDING"


# ============================================================
# SPEED CALCULATION
# ============================================================

def calculate_speed(vehicle, current_position, current_time):

    history = vehicle["history"]

    if len(history) < 2:

        return 0

    # Use a previous stable position instead of
    # only the immediately previous frame.
    previous_position, previous_time = history[-2]

    distance_pixels = np.sqrt(
        (
            current_position[0]
            - previous_position[0]
        ) ** 2
        +
        (
            current_position[1]
            - previous_position[1]
        ) ** 2
    )

    distance_meters = (
        distance_pixels
        * PIXEL_TO_METER_RATIO
    )

    time_seconds = (
        current_time
        - previous_time
    )

    if time_seconds <= 0:

        return 0

    speed_ms = (
        distance_meters
        / time_seconds
    )

    speed_kmh = speed_ms * 3.6

    # Remove unrealistic spikes
    if speed_kmh > MAX_SPEED:

        return vehicle.get(
            "speed",
            0
        )

    return speed_kmh


# ============================================================
# PROCESS FRAME
# ============================================================

def process_frame(frame):

    global highest_speed
    global total_detected

    current_time = time.time()

    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        classes=list(VEHICLE_CLASSES.keys()),
        conf=0.40,
        iou=0.50,
        verbose=False
    )

    if not results:

        return frame

    result = results[0]

    boxes = result.boxes

    detected_ids = set()

    # --------------------------------------------------------
    # DETECTIONS
    # --------------------------------------------------------

    if boxes is not None and len(boxes) > 0:

        for box in boxes:

            if box.id is None:

                continue

            track_id = int(
                box.id.item()
            )

            class_id = int(
                box.cls.item()
            )

            confidence = float(
                box.conf.item()
            )

            if class_id not in VEHICLE_CLASSES:

                continue

            vehicle_type = VEHICLE_CLASSES[
                class_id
            ]

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )

            center = (
                center_x,
                center_y
            )

            detected_ids.add(track_id)

            # ------------------------------------------------
            # New vehicle
            # ------------------------------------------------

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

                        "speed": 0,

                        "last_seen":
                            current_time,

                        "max_speed": 0
                    }

                    total_detected += 1

                vehicle = vehicles[
                    track_id
                ]

                # ------------------------------------------------
                # Update position
                # ------------------------------------------------

                vehicle["position"] = center

                vehicle["type"] = vehicle_type

                vehicle["history"].append(
                    (
                        center,
                        current_time
                    )
                )

                # Keep last 10 positions
                if len(
                    vehicle["history"]
                ) > 10:

                    vehicle["history"] = \
                        vehicle[
                            "history"
                        ][-10:]

                vehicle["last_seen"] = \
                    current_time

                # ------------------------------------------------
                # Calculate speed
                # ------------------------------------------------

                speed = calculate_speed(
                    vehicle,
                    center,
                    current_time
                )

                # Smooth speed
                old_speed = vehicle[
                    "speed"
                ]

                if old_speed > 0:

                    speed = (
                        old_speed * 0.65
                        +
                        speed * 0.35
                    )

                vehicle["speed"] = speed

                if speed > vehicle[
                    "max_speed"
                ]:

                    vehicle[
                        "max_speed"
                    ] = speed

                if speed > highest_speed:

                    highest_speed = speed

            # ------------------------------------------------
            # DRAW
            # ------------------------------------------------

            color = get_speed_color(
                speed
            )

            status = get_speed_status(
                speed
            )

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # ------------------------------------------------
            # Label background
            # ------------------------------------------------

            label = (
                f"#{track_id} "
                f"{vehicle_type}"
            )

            speed_label = (
                f"{int(speed)} km/h"
            )

            status_label = status

            # Label positions
            label_y = max(
                y1 - 10,
                25
            )

            # Vehicle name
            cv2.putText(
                frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

            # Speed
            cv2.putText(
                frame,
                speed_label,
                (
                    x1,
                    y2 + 25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

            # Status
            cv2.putText(
                frame,
                status_label,
                (
                    x1,
                    y2 + 50
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    # ========================================================
    # REMOVE OLD VEHICLES
    # ========================================================

    with lock:

        expired = []

        for vid, vehicle in vehicles.items():

            if (
                current_time
                - vehicle["last_seen"]
                > 2.5
            ):

                expired.append(
                    vid
                )

        for vid in expired:

            vehicle = vehicles[
                vid
            ]

            # Save final history
            vehicle_history.append({

                "id": vehicle["id"],

                "type": vehicle["type"],

                "max_speed": round(
                    vehicle["max_speed"],
                    1
                ),

                "time": time.strftime(
                    "%H:%M:%S"
                )
            })

            del vehicles[vid]

    # ========================================================
    # DASHBOARD OVERLAY
    # ========================================================

    overlay_height = 110

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (330, overlay_height),
        (10, 10, 10),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.80,
        frame,
        0.20,
        0,
        frame
    )

    cv2.putText(
        frame,
        "REAL-TIME VEHICLE TRACKER",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Vehicles: {len(vehicles)}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Highest: {int(highest_speed)} km/h",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "YOLO VEHICLE DETECTION",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (0, 255, 0),
        1
    )

    return frame


# ============================================================
# FRAME PROCESSOR
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

            print(
                "Processing error:",
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
        (100, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        blank,
        "Open /phone on your phone",
        (130, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (170, 170, 170),
        1
    )

    _, blank_encoded = cv2.imencode(
        ".jpg",
        blank
    )

    blank_bytes = \
        blank_encoded.tobytes()

    while True:

        with lock:

            current_frame = \
                latest_processed_frame

        if current_frame is None:

            frame_bytes = blank_bytes

        else:

            frame_bytes = current_frame

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
# PHONE CAMERA UPLOAD
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
                "status": "error"
            }), 400

        file = request.files[
            "image"
        ].read()

        npimg = np.frombuffer(
            file,
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

        # Standard processing size
        frame = cv2.resize(
            frame,
            (640, 480)
        )

        if frame_queue.full():

            try:

                frame_queue.get_nowait()

            except queue.Empty:

                pass

        frame_queue.put_nowait(
            frame
        )

        camera_active = True

        last_frame_time = \
            time.time()

        return jsonify({
            "status": "success"
        })

    except Exception as e:

        print(
            "Upload error:",
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
        time.time()
        - last_frame_time
        > 3
    ):

        camera_active = False

    with lock:

        current_vehicles = []

        for vehicle in vehicles.values():

            current_vehicles.append({

                "id": vehicle["id"],

                "type": vehicle["type"],

                "speed": round(
                    vehicle["speed"],
                    1
                ),

                "max_speed": round(
                    vehicle["max_speed"],
                    1
                ),

                "status":
                    get_speed_status(
                        vehicle["speed"]
                    )
            })

        history = \
            vehicle_history[-20:]

    return jsonify({

        "camera_active":
            camera_active,

        "vehicles":
            current_vehicles,

        "vehicle_count":
            len(current_vehicles),

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
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    local_ip = get_local_ip()

    print("\n")
    print("=" * 60)
    print("       REAL-TIME VEHICLE SPEED TRACKER V2")
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
        "\nMake sure both devices are"
    )

    print(
        "connected to the SAME Wi-Fi."
    )

    print("=" * 60)
    print("\n")

    app.run(
        host=HOST,
        port=PORT,
        threaded=True
    )
