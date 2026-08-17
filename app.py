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

# n = much faster on CPU. Change to yolo11s.pt if your laptop
# has enough CPU/GPU and you want a little more detection accuracy.
MODEL_NAME = "yolo11n.pt"

# Keep the analysis/display frame in 16:9 landscape.
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# Phone -> server upload target.
PHONE_FPS = 18

# Laptop MJPEG output.
OUTPUT_FPS = 20

# YOLO is intentionally not run on every display frame.
# The last detector result is kept on screen between detections.
DETECTION_INTERVAL = 0.10

JPEG_QUALITY = 72

# Real-world distance between line A and line B.
# IMPORTANT: measure this on the road.
CALIBRATION_DISTANCE_METERS = 10.0

NORMAL_SPEED = 40
MODERATE_SPEED = 60
MAX_SPEED = 200

# COCO classes used by the pretrained model:
# 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# Lower confidence helps motorcycles/bikes survive small/blurred frames.
CLASS_CONFIDENCE = {
    1: 0.15,
    2: 0.20,
    3: 0.12,
    5: 0.20,
    7: 0.20,
}

# ============================================================
# GLOBAL STATE
# ============================================================

latest_raw_frame = None
camera_active = False
last_camera_frame_time = 0.0

vehicles = {}
vehicle_history = []

highest_speed = 0.0
total_detected = 0

lock = threading.Lock()

# Default calibration lines. These are only a starting point.
# The laptop UI lets you replace them with road-specific lines.
line_a = [(40, 150), (600, 150)]
line_b = [(40, 245), (600, 245)]

# ============================================================
# MODEL
# ============================================================

print("=" * 60)
print("Loading YOLO model:", MODEL_NAME)
print("=" * 60)

model = YOLO(MODEL_NAME)

print("YOLO model loaded.")
print("=" * 60)


# ============================================================
# HELPERS
# ============================================================

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def point_distance(p1, p2):
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )


def clamp_point(p):
    return (
        max(0, min(FRAME_WIDTH - 1, int(p[0]))),
        max(0, min(FRAME_HEIGHT - 1, int(p[1])))
    )


def cross(a, b, c):
    # Orientation of a->b->c
    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def on_segment(a, b, p, eps=1e-6):
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and
        min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def segments_intersect(a, b, c, d):
    """
    True when movement segment A-B actually crosses calibration
    segment C-D. This fixes the old 'only compare Y' limitation.
    """
    o1 = cross(a, b, c)
    o2 = cross(a, b, d)
    o3 = cross(c, d, a)
    o4 = cross(c, d, b)

    eps = 1e-7

    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
       ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True

    if abs(o1) <= eps and on_segment(a, b, c):
        return True
    if abs(o2) <= eps and on_segment(a, b, d):
        return True
    if abs(o3) <= eps and on_segment(c, d, a):
        return True
    if abs(o4) <= eps and on_segment(c, d, b):
        return True

    return False


def get_speed_color(speed, moving, measured):
    if not measured:
        return (180, 180, 180)

    if not moving:
        return (160, 160, 160)

    if speed < NORMAL_SPEED:
        return (0, 255, 0)
    elif speed < MODERATE_SPEED:
        return (0, 255, 255)
    return (0, 0, 255)


def get_speed_status(speed, moving, measured):
    if not measured:
        return "MEASURING"

    if not moving:
        return "PARKED"

    if speed < NORMAL_SPEED:
        return "NORMAL"
    elif speed < MODERATE_SPEED:
        return "MODERATE"
    return "SPEEDING"


def line_crossed(previous_point, current_point, line):
    if previous_point is None:
        return False

    if line is None or len(line) != 2:
        return False

    return segments_intersect(
        previous_point,
        current_point,
        tuple(line[0]),
        tuple(line[1])
    )


def calculate_line_speed(vehicle):
    a_time = vehicle.get("line_a_time")
    b_time = vehicle.get("line_b_time")

    if a_time is None or b_time is None:
        return 0.0

    delta = abs(b_time - a_time)

    if delta < 0.08:
        return 0.0

    speed_ms = CALIBRATION_DISTANCE_METERS / delta
    speed_kmh = speed_ms * 3.6

    if speed_kmh > MAX_SPEED:
        return 0.0

    return speed_kmh


def vehicle_is_moving(vehicle):
    history = vehicle.get("history", [])

    if len(history) < 4:
        return False

    # Compare a short window rather than a single detector jump.
    p1 = history[-1][0]
    p2 = history[-4][0]

    return point_distance(p1, p2) >= 5.0


# ============================================================
# DETECTION
# ============================================================

def detect_vehicles(frame):
    global highest_speed
    global total_detected

    now = time.time()

    try:
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(VEHICLE_CLASSES.keys()),
            conf=0.12,
            iou=0.45,
            imgsz=640,
            max_det=35,
            verbose=False
        )
    except Exception as error:
        print("YOLO ERROR:", error)
        return

    if not results:
        return

    result = results[0]
    boxes = result.boxes

    if boxes is None:
        return

    current_ids = set()

    for box in boxes:
        try:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
        except Exception:
            continue

        if class_id not in VEHICLE_CLASSES:
            continue

        if confidence < CLASS_CONFIDENCE.get(class_id, 0.15):
            continue

        if box.id is None:
            continue

        try:
            track_id = int(box.id[0].item())
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )
        except Exception:
            continue

        x1 = max(0, min(FRAME_WIDTH - 1, x1))
        y1 = max(0, min(FRAME_HEIGHT - 1, y1))
        x2 = max(0, min(FRAME_WIDTH - 1, x2))
        y2 = max(0, min(FRAME_HEIGHT - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        # Reject boxes that cover almost the entire image.
        area_ratio = (
            ((x2 - x1) * (y2 - y1))
            / float(FRAME_WIDTH * FRAME_HEIGHT)
        )

        if area_ratio > 0.70:
            continue

        # Bottom-center is the best point for road-speed line crossing.
        ground_point = (
            int((x1 + x2) / 2),
            int(y2)
        )

        current_ids.add(track_id)

        if track_id not in vehicles:
            vehicles[track_id] = {
                "id": track_id,
                "type": VEHICLE_CLASSES[class_id],
                "class_id": class_id,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "position": ground_point,
                "previous_position": None,
                "history": [],
                "speed": 0.0,
                "max_speed": 0.0,
                "moving": False,
                "line_a_time": None,
                "line_b_time": None,
                "speed_locked": False,
                "last_seen": now,
                "first_seen": now,
            }
            total_detected += 1

        vehicle = vehicles[track_id]

        previous_position = vehicle["position"]

        vehicle["previous_position"] = previous_position
        vehicle["position"] = ground_point
        vehicle["box"] = (x1, y1, x2, y2)
        vehicle["type"] = VEHICLE_CLASSES[class_id]
        vehicle["class_id"] = class_id
        vehicle["confidence"] = confidence
        vehicle["last_seen"] = now

        vehicle["history"].append(
            (ground_point, now)
        )

        if len(vehicle["history"]) > 30:
            vehicle["history"] = vehicle["history"][-30:]

        vehicle["moving"] = vehicle_is_moving(vehicle)

        # --------------------------------------------------------
        # LINE A / LINE B CROSSING
        # --------------------------------------------------------

        with lock:
            current_line_a = list(line_a)
            current_line_b = list(line_b)

        if (
            vehicle["line_a_time"] is None
            and
            line_crossed(
                previous_position,
                ground_point,
                current_line_a
            )
        ):
            vehicle["line_a_time"] = now

        if (
            vehicle["line_b_time"] is None
            and
            line_crossed(
                previous_position,
                ground_point,
                current_line_b
            )
        ):
            vehicle["line_b_time"] = now

        # --------------------------------------------------------
        # SPEED AFTER BOTH LINES
        # --------------------------------------------------------

        if (
            vehicle["line_a_time"] is not None
            and
            vehicle["line_b_time"] is not None
            and
            not vehicle["speed_locked"]
        ):
            calculated_speed = calculate_line_speed(vehicle)

            if calculated_speed > 0:
                vehicle["speed"] = calculated_speed
                vehicle["max_speed"] = calculated_speed
                vehicle["speed_locked"] = True

                if calculated_speed > highest_speed:
                    highest_speed = calculated_speed

        # Parked / not-yet-measured vehicle stays at 0.
        if not vehicle["speed_locked"]:
            vehicle["speed"] = 0.0


def cleanup_vehicles():
    now = time.time()
    remove_ids = []

    with lock:
        for vehicle_id, vehicle in vehicles.items():
            if now - vehicle["last_seen"] > 2.0:
                remove_ids.append(vehicle_id)

        for vehicle_id in remove_ids:
            vehicle = vehicles[vehicle_id]

            if vehicle["speed_locked"]:
                vehicle_history.append({
                    "id": vehicle["id"],
                    "type": vehicle["type"],
                    "speed": round(vehicle["speed"], 1),
                    "status": get_speed_status(
                        vehicle["speed"],
                        vehicle["moving"],
                        True
                    ),
                    "time": time.strftime("%H:%M:%S")
                })

            del vehicles[vehicle_id]

        if len(vehicle_history) > 100:
            del vehicle_history[:-100]


# ============================================================
# DRAW
# ============================================================

def draw_line(output, line, label, color):
    if not line or len(line) != 2:
        return

    p1 = tuple(map(int, line[0]))
    p2 = tuple(map(int, line[1]))

    cv2.line(
        output,
        p1,
        p2,
        color,
        3,
        cv2.LINE_AA
    )

    # Endpoint circles make the line easy to understand.
    cv2.circle(output, p1, 6, color, -1)
    cv2.circle(output, p2, 6, color, -1)

    cv2.putText(
        output,
        label,
        (
            p1[0] + 8,
            max(20, p1[1] - 8)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA
    )


def draw_overlay(frame):
    output = frame.copy()

    with lock:
        current_line_a = list(line_a)
        current_line_b = list(line_b)
        vehicle_snapshot = [
            dict(v)
            for v in vehicles.values()
        ]
        current_highest = highest_speed
        current_total = total_detected

    # Calibration lines
    draw_line(
        output,
        current_line_a,
        "A",
        (255, 180, 0)
    )

    draw_line(
        output,
        current_line_b,
        "B",
        (255, 0, 255)
    )

    # Vehicle boxes
    for vehicle in vehicle_snapshot:
        x1, y1, x2, y2 = vehicle["box"]

        speed = vehicle["speed"]
        moving = vehicle["moving"]
        measured = vehicle["speed_locked"]

        color = get_speed_color(
            speed,
            moving,
            measured
        )

        status = get_speed_status(
            speed,
            moving,
            measured
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2,
            cv2.LINE_AA
        )

        label = (
            f'#{vehicle["id"]} '
            f'{vehicle["type"]} '
            f'{vehicle["confidence"]:.0%}'
        )

        label_y = max(20, y1 - 8)

        cv2.putText(
            output,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA
        )

        speed_text = (
            f'{int(round(speed))} km/h'
            if measured
            else '0 km/h'
        )

        speed_y = min(
            output.shape[0] - 30,
            y2 + 22
        )

        cv2.putText(
            output,
            speed_text,
            (x1, speed_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA
        )

        status_y = min(
            output.shape[0] - 8,
            y2 + 43
        )

        cv2.putText(
            output,
            status,
            (x1, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            2,
            cv2.LINE_AA
        )

        px, py = vehicle["position"]

        cv2.circle(
            output,
            (px, py),
            4,
            color,
            -1
        )

    # Dashboard
    overlay = output.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (365, 145),
        (0, 0, 0),
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
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Live Vehicles: {len(vehicle_snapshot)}",
        (20, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Highest Speed: {int(round(current_highest))} km/h",
        (20, 87),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Total Detected: {current_total}",
        (20, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"Distance A-B: {CALIBRATION_DISTANCE_METERS:g} m",
        (20, 137),
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
    last_detection = 0.0

    while True:
        with lock:
            frame = (
                latest_raw_frame.copy()
                if latest_raw_frame is not None
                else None
            )

        if frame is None:
            time.sleep(0.02)
            continue

        now = time.time()

        if now - last_detection >= DETECTION_INTERVAL:
            last_detection = now

            detect_vehicles(frame)
            cleanup_vehicles()

        time.sleep(0.005)


# ============================================================
# VIDEO STREAM
# ============================================================

def encode_jpeg(frame):
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            JPEG_QUALITY
        ]
    )

    return encoded.tobytes() if ok else b""


def generate_frames():
    blank = np.zeros(
        (FRAME_HEIGHT, FRAME_WIDTH, 3),
        dtype=np.uint8
    )

    cv2.putText(
        blank,
        "WAITING FOR PHONE CAMERA...",
        (145, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )

    cv2.putText(
        blank,
        "Open /phone on the phone",
        (190, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1
    )

    blank_bytes = encode_jpeg(blank)

    while True:
        with lock:
            frame = (
                latest_raw_frame.copy()
                if latest_raw_frame is not None
                else None
            )

        if frame is None:
            frame_bytes = blank_bytes
        else:
            output = draw_overlay(frame)
            frame_bytes = encode_jpeg(output)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Pragma: no-cache\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

        time.sleep(1.0 / OUTPUT_FPS)


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/phone")
def phone():
    return render_template("phone.html")


@app.route("/monitor")
def monitor():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache"
        }
    )


@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    global latest_raw_frame
    global camera_active
    global last_camera_frame_time

    try:
        if "image" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image"
            }), 400

        image_data = request.files["image"].read()

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

        # Resize while keeping the required 16:9 output.
        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        with lock:
            latest_raw_frame = frame
            camera_active = True
            last_camera_frame_time = time.time()

        return jsonify({"status": "success"})

    except Exception as error:
        print("UPLOAD ERROR:", error)

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500


@app.route("/set_lines", methods=["POST"])
def set_lines():
    global line_a
    global line_b
    global CALIBRATION_DISTANCE_METERS

    try:
        data = request.get_json(force=True)

        new_a = data.get("line_a")
        new_b = data.get("line_b")
        distance = data.get(
            "distance",
            CALIBRATION_DISTANCE_METERS
        )

        def validate_line(value):
            return (
                isinstance(value, list)
                and len(value) == 2
                and all(
                    isinstance(p, list)
                    and len(p) == 2
                    for p in value
                )
            )

        if not validate_line(new_a):
            return jsonify({
                "status": "error",
                "message": "Invalid line A"
            }), 400

        if not validate_line(new_b):
            return jsonify({
                "status": "error",
                "message": "Invalid line B"
            }), 400

        distance = float(distance)

        if distance <= 0 or distance > 1000:
            return jsonify({
                "status": "error",
                "message": "Distance must be between 0 and 1000 meters"
            }), 400

        with lock:
            line_a = [
                clamp_point(new_a[0]),
                clamp_point(new_a[1])
            ]

            line_b = [
                clamp_point(new_b[0]),
                clamp_point(new_b[1])
            ]

            CALIBRATION_DISTANCE_METERS = distance

            # Existing vehicles must be measured again with
            # the new calibration.
            for vehicle in vehicles.values():
                vehicle["line_a_time"] = None
                vehicle["line_b_time"] = None
                vehicle["speed_locked"] = False
                vehicle["speed"] = 0.0

        return jsonify({
            "status": "success",
            "line_a": line_a,
            "line_b": line_b,
            "distance": CALIBRATION_DISTANCE_METERS
        })

    except Exception as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400


@app.route("/reset_tracking", methods=["POST"])
def reset_tracking():
    global vehicles
    global vehicle_history
    global highest_speed
    global total_detected

    with lock:
        vehicles.clear()
        vehicle_history.clear()
        highest_speed = 0.0
        total_detected = 0

    return jsonify({"status": "success"})


@app.route("/status")
def status():
    global camera_active

    if time.time() - last_camera_frame_time > 3:
        camera_active = False

    with lock:
        live_vehicles = []

        for vehicle in vehicles.values():
            live_vehicles.append({
                "id": vehicle["id"],
                "type": vehicle["type"],
                "speed": round(vehicle["speed"], 1),
                "max_speed": round(
                    vehicle["max_speed"], 1
                ),
                "status": get_speed_status(
                    vehicle["speed"],
                    vehicle["moving"],
                    vehicle["speed_locked"]
                ),
                "confidence": round(
                    vehicle["confidence"], 2
                ),
                "measured": vehicle["speed_locked"]
            })

        history = vehicle_history[-30:]

        current_lines = {
            "a": line_a,
            "b": line_b
        }

    moving_speeds = [
        v["speed"]
        for v in vehicles.values()
        if v["speed_locked"] and v["speed"] > 0
    ]

    average_speed = (
        sum(moving_speeds) / len(moving_speeds)
        if moving_speeds
        else 0
    )

    return jsonify({
        "camera_active": camera_active,
        "vehicles": live_vehicles,
        "vehicle_count": len(live_vehicles),
        "highest_speed": round(highest_speed, 1),
        "average_speed": round(average_speed, 1),
        "total_detected": total_detected,
        "history": history,
        "calibration_distance": CALIBRATION_DISTANCE_METERS,
        "lines": current_lines
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    local_ip = get_local_ip()

    print()
    print("=" * 60)
    print("       VEHICLE SPEED TRACKER V3")
    print("=" * 60)
    print()
    print("PHONE CAMERA:")
    print(f"http://{local_ip}:5000/phone")
    print()
    print("LAPTOP MONITOR:")
    print(f"http://{local_ip}:5000/monitor")
    print()
    print("Landscape camera recommended.")
    print("Measure the real road distance between A and B.")
    print("=" * 60)
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
