import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
import time
import threading
import queue
import socket

app = Flask(__name__)

# ============================================================
# GLOBAL STATE
# ============================================================

vehicles = {}
vehicle_counter = 0

lock = threading.Lock()

# Speed settings
PIXEL_TO_METER_RATIO = 0.05
NORMAL_SPEED = 40
MODERATE_SPEED = 60

# Frames coming from phone
frame_queue = queue.Queue(maxsize=3)

# Latest processed frame for laptop monitor
latest_processed_frame = None

# Camera connection information
camera_active = False
last_frame_time = 0

# Statistics
total_detected = 0
highest_speed = 0


# ============================================================
# NETWORK
# ============================================================

def get_local_ip():
    """Get laptop's local IP address."""

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip

    except Exception:
        return "127.0.0.1"


# ============================================================
# SPEED CALCULATION
# ============================================================

def calculate_speed(vehicle_data, current_position, current_time):
    """Calculate vehicle speed using pixel movement."""

    if len(vehicle_data["history"]) < 2:
        return 0

    last_pos, last_time = vehicle_data["history"][-2]

    distance_pixels = np.sqrt(
        (current_position[0] - last_pos[0]) ** 2 +
        (current_position[1] - last_pos[1]) ** 2
    )

    distance_meters = distance_pixels * PIXEL_TO_METER_RATIO

    time_seconds = current_time - last_time

    if time_seconds <= 0:
        return 0

    speed_ms = distance_meters / time_seconds
    speed_kmh = speed_ms * 3.6

    return min(speed_kmh, 200)


# ============================================================
# SPEED COLOR
# ============================================================

def get_box_color(speed):

    if speed < NORMAL_SPEED:
        return (0, 255, 0)       # Green

    elif speed < MODERATE_SPEED:
        return (0, 255, 255)     # Yellow

    else:
        return (0, 0, 255)       # Red


# ============================================================
# VEHICLE DETECTION
# ============================================================

def detect_vehicle(frame):
    """
    Experimental vehicle detection using OpenCV contours.
    This is NOT AI/YOLO detection.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray,
        (21, 21),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    vehicles_found = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if 1000 < area < 50000:

            x, y, w, h = cv2.boundingRect(contour)

            aspect_ratio = w / float(h)

            if 0.5 < aspect_ratio < 3.0:

                vehicles_found.append(
                    (x, y, w, h)
                )

    return vehicles_found


# ============================================================
# FRAME PROCESSING
# ============================================================

def process_frame(frame):

    global vehicle_counter
    global total_detected
    global highest_speed

    current_time = time.time()

    vehicles_found = detect_vehicle(frame)

    with lock:

        # ----------------------------------------------------
        # Remove old vehicles
        # ----------------------------------------------------

        vehicles_to_remove = []

        for vid, vehicle in vehicles.items():

            if current_time - vehicle["last_seen"] > 1.5:

                vehicles_to_remove.append(vid)

        for vid in vehicles_to_remove:

            del vehicles[vid]

        # ----------------------------------------------------
        # Match detected objects
        # ----------------------------------------------------

        used_vehicle_ids = set()

        for x, y, w, h in vehicles_found:

            center = (
                x + w // 2,
                y + h // 2
            )

            matched_vid = None

            smallest_distance = float("inf")

            for vid, vehicle in vehicles.items():

                if vid in used_vehicle_ids:
                    continue

                last_position = vehicle["position"]

                distance = np.sqrt(
                    (center[0] - last_position[0]) ** 2 +
                    (center[1] - last_position[1]) ** 2
                )

                if distance < 80 and distance < smallest_distance:

                    smallest_distance = distance
                    matched_vid = vid

            # ------------------------------------------------
            # New vehicle
            # ------------------------------------------------

            if matched_vid is None:

                vehicle_counter += 1

                matched_vid = vehicle_counter

                vehicles[matched_vid] = {

                    "id": matched_vid,

                    "position": center,

                    "history": [
                        (center, current_time)
                    ],

                    "last_seen": current_time,

                    "speed": 0
                }

                total_detected += 1

            # ------------------------------------------------
            # Update vehicle
            # ------------------------------------------------

            vehicle = vehicles[matched_vid]

            used_vehicle_ids.add(matched_vid)

            vehicle["position"] = center

            vehicle["history"].append(
                (center, current_time)
            )

            if len(vehicle["history"]) > 10:

                vehicle["history"] = \
                    vehicle["history"][-10:]

            vehicle["last_seen"] = current_time

            # ------------------------------------------------
            # Calculate speed
            # ------------------------------------------------

            speed = calculate_speed(
                vehicle,
                center,
                current_time
            )

            vehicle["speed"] = speed

            if speed > highest_speed:

                highest_speed = speed

        # ----------------------------------------------------
        # Draw vehicle information
        # ----------------------------------------------------

        for vid, vehicle in vehicles.items():

            speed = vehicle["speed"]

            color = get_box_color(speed)

            cx, cy = vehicle["position"]

            # Dynamic box
            x = cx - 40
            y = cy - 30

            w = 80
            h = 60

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            # Vehicle ID
            cv2.putText(
                frame,
                f"CAR #{vehicle['id']}",
                (x, y - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

            # Speed
            cv2.putText(
                frame,
                f"{int(speed)} km/h",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # ========================================================
    # DASHBOARD OVERLAY
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (300, 120),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        "REAL-TIME SPEED TRACKER",
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
        "PHONE CAMERA CONNECTED",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1
    )

    return frame


# ============================================================
# PROCESSOR THREAD
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

            processed = process_frame(frame)

            success, encoded = cv2.imencode(
                ".jpg",
                processed,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    80
                ]
            )

            if success:

                with lock:

                    latest_processed_frame = \
                        encoded.tobytes()

        except Exception as e:

            print(
                "Frame processing error:",
                e
            )


# Start processing thread
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
        (110, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        blank,
        "Open /phone on your phone",
        (140, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        1
    )

    success, encoded_blank = cv2.imencode(
        ".jpg",
        blank
    )

    blank_bytes = encoded_blank.tobytes()

    while True:

        with lock:

            frame = latest_processed_frame

        if frame is None:

            frame_bytes = blank_bytes

        else:

            frame_bytes = frame

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
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
        mimetype="multipart/x-mixed-replace; boundary=frame"
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

        file = request.files["image"].read()

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
                "status": "error",
                "message": "Invalid image"
            }), 400

        # Resize for performance
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
        last_frame_time = time.time()

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

    if time.time() - last_frame_time > 3:

        camera_active = False

    with lock:

        vehicle_count = len(vehicles)

        current_highest = highest_speed

    return jsonify({

        "camera_active": camera_active,

        "vehicles": vehicle_count,

        "highest_speed":
            round(current_highest, 1),

        "total_detected":
            total_detected
    })


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    local_ip = get_local_ip()

    print("\n")
    print("=" * 55)
    print("       REAL-TIME VEHICLE SPEED TRACKER")
    print("=" * 55)

    print(
        f"\n📱 PHONE CAMERA:"
        f"\nhttp://{local_ip}:5000/phone"
    )

    print(
        f"\n💻 LAPTOP MONITOR:"
        f"\nhttp://{local_ip}:5000/monitor"
    )

    print(
        f"\n🏠 HOME:"
        f"\nhttp://{local_ip}:5000/"
    )

    print("\nMake sure phone and laptop are")
    print("connected to the SAME Wi-Fi.")
    print("=" * 55)
    print("\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )
