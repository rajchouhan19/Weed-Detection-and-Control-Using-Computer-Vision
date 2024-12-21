from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load your YOLO model
model = YOLO("D:/newwww/runs/detect/train6/weights/best.pt")

# Global variables
video_source = None  # No source initially
is_running = False  # Detection is off by default

def generate_frames(source):
    global is_running
    cap = cv2.VideoCapture(source)
    try:
        while is_running:
            success, frame = cap.read()  # Read the camera frame
            if not success:
                break
            else:
                # Run prediction on the current frame
                results = model.predict(source=frame, stream=True)

                # Process each result and add bounding boxes
                for result in results:
                    frame = result.plot()  # Overlay bounding boxes and labels on the frame

                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                # Yield the frame in byte format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        # Ensure the camera is released
        cap.release()

@app.route('/', methods=['GET', 'POST'])
def index2():
    global video_source, is_running
    if request.method == 'POST':
        if 'video' in request.files:
            # Save uploaded video
            video_file = request.files['video']
            video_path = f"./static/{video_file.filename}"
            os.makedirs('./static', exist_ok=True)
            video_file.save(video_path)
            video_source = video_path
            is_running = True
        elif 'real_time' in request.form:
            # Start real-time detection
            video_source = 0  # Webcam
            is_running = True
        elif 'stop' in request.form:
            # Stop detection
            is_running = False
    return render_template('index3.html')

@app.route('/video_feed')
def video_feed():
    global video_source
    if video_source is not None:
        return Response(generate_frames(video_source), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "No video source selected.", 400

if __name__ == "__main__":
    app.run(debug=True)
