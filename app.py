import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
import time
import threading
import queue
from collections import deque
import socket

app = Flask(__name__)

# Vehicle tracking data
vehicles = {}
vehicle_counter = 0
lock = threading.Lock()

# Speed calculation settings
PIXEL_TO_METER_RATIO = 0.05  # Calibration value - adjust this
NORMAL_SPEED = 40
MODERATE_SPEED = 60

# Queue for video frames from phone
frame_queue = queue.Queue(maxsize=5)
camera_active = False

def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def calculate_speed(vehicle_data, current_position, current_time):
    """Calculate speed of a vehicle"""
    if len(vehicle_data['history']) < 2:
        return 0
    
    last_pos = vehicle_data['history'][-2][0]
    last_time = vehicle_data['history'][-2][1]
    
    dist_pixels = np.sqrt(
        (current_position[0] - last_pos[0])**2 + 
        (current_position[1] - last_pos[1])**2
    )
    
    dist_meters = dist_pixels * PIXEL_TO_METER_RATIO
    time_sec = (current_time - last_time) / 1000
    
    if time_sec <= 0:
        return 0
    
    speed_ms = dist_meters / time_sec
    speed_kmh = speed_ms * 3.6
    
    return min(speed_kmh, 200)  # Cap at 200 km/h

def get_box_color(speed):
    """Return color based on speed"""
    if speed < NORMAL_SPEED:
        return (0, 255, 0)  # Green
    elif speed < MODERATE_SPEED:
        return (255, 255, 0)  # Yellow
    else:
        return (0, 0, 255)  # Red

def detect_vehicle(frame):
    """Simple vehicle detection using contour detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    vehicles_found = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area (adjust threshold based on your camera)
        if 1000 < area < 50000:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (vehicles are wider than tall)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 3.0:
                vehicles_found.append((x, y, w, h))
    
    return vehicles_found

def process_frame(frame):
    """Process frame: detect vehicles, calculate speeds, draw boxes"""
    vehicles_found = detect_vehicle(frame)
    current_time = cv2.getTickCount()
    
    # Update existing vehicles or create new ones
    with lock:
        # Remove vehicles that are no longer in frame
        vehicles_to_remove = []
        for vid, vdata in vehicles.items():
            if vdata['last_seen'] < current_time - 1000:  # Not seen in 1 second
                vehicles_to_remove.append(vid)
        
        for vid in vehicles_to_remove:
            del vehicles[vid]
        
        # Update or create vehicles
        for i, (x, y, w, h) in enumerate(vehicles_found):
            center = (x + w // 2, y + h // 2)
            
            # Find matching vehicle or create new
            matched_vid = None
            for vid, vdata in vehicles.items():
                last_pos = vdata['position']
                dist = np.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                if dist < 50:  # Match threshold
                    matched_vid = vid
                    break
            
            if matched_vid:
                vehicle = vehicles[matched_vid]
            else:
                vehicle_counter += 1
                matched_vid = vehicle_counter
                vehicles[matched_vid] = {
                    'id': matched_vid,
                    'position': center,
                    'history': [(center, current_time)],
                    'last_seen': current_time,
                    'speed': 0
                }
            
            vehicle = vehicles[matched_vid]
            vehicle['position'] = center
            vehicle['history'].append((center, current_time))
            if len(vehicle['history']) > 10:
                vehicle['history'] = vehicle['history'][-10:]
            vehicle['last_seen'] = current_time
            
            # Calculate speed
            speed = calculate_speed(vehicle, center, current_time)
            vehicle['speed'] = speed
    
    # Draw on frame
    with lock:
        for vid, vehicle in vehicles.items():
            color = get_box_color(vehicle['speed'])
            
            # Get last position
            if len(vehicle['history']) > 1:
                x = vehicle['position'][0] - 25
                y = vehicle['position'][1] - 25
                w, h = 50, 50
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Speed text
                speed_text = f"{int(vehicle['speed'])} km/h"
                cv2.putText(frame, speed_text, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # ID text
                id_text = f"ID: {vehicle['id']}"
                cv2.putText(frame, id_text, (x, y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

# ... (previous code continues) ...

def generate_frames():
    """Generate video frames for the laptop view"""
    global camera_active, frame_queue
    
    # This loop will try to get frames from the phone's stream
    while True:
        if not frame_queue.empty():
            # Get frame from phone's camera
            frame = frame_queue.get()
            
            # Process the frame for vehicle detection and speed
            processed_frame = process_frame(frame)
            
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If no frame from phone, wait a bit
            time.sleep(0.01)

@app.route('/')
def index():
    """Video streaming home page for laptop view"""
    return render_template('index.html')

@app.route('/phone_view')
def phone_view():
    """Page for the phone to capture and send video"""
    return render_template('phone.html')

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """Endpoint to receive frames from the phone"""
    global frame_queue
    try:
        # Get image data from the request
        file = request.files['image'].read()
        # Decode image
        npimg = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if frame is not None:
            # Add frame to queue for processing
            if frame_queue.full():
                frame_queue.get()  # Remove oldest frame if queue is full
            frame_queue.put(frame)
        
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error receiving frame: {e}")
        return jsonify({"status": "error"}), 500

if __name__ == '__main__':
    local_ip = get_local_ip()
    print(f"Starting server...")
    print(f"Open this URL on your LAPTOP for live view: http://{local_ip}:5000/")
    print(f"Open this URL on your PHONE to send camera stream: http://{local_ip}:5000/phone_view")
    
    app.run(host='0.0.0.0', port=5000, threaded=True)
