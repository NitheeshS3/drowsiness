#!/usr/bin/env python3
from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import socket, random, os

app = Flask(__name__)

# -------- Load Model --------
MODEL_PATH = "model/my_model.h5"
if not os.path.exists(MODEL_PATH):
    raise SystemExit("❌ Model not found. Place trained model in model/my_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# -------- Settings --------
IMG_SIZE = 145
SEQ_LEN = 10

frame_buffer = deque(maxlen=SEQ_LEN)
smooth_buffer = deque(maxlen=8)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    camera.open(0)

# Global state shared with frontend
drowsy_state = {"status": "alert"}


def preprocess(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame.astype(np.float32) / 255.0


def generate_frames():
    drowsy_count = 0
    CLOSED_FRAMES_THRESHOLD = 45  # Sustained 3–4 sec eye closure required

    while True:
        success, frame = camera.read()
        if not success:
            break

        processed = preprocess(frame)
        frame_buffer.append(processed)

        if len(frame_buffer) == SEQ_LEN:
            inp = np.expand_dims(np.array(frame_buffer), axis=0)
            pred = float(model.predict(inp, verbose=0)[0][0])

            smooth_buffer.append(pred)
            avg = np.mean(smooth_buffer)

            if avg > 0.5:
                drowsy_count += 1
            else:
                drowsy_count -= 1
            
            drowsy_count = max(0, min(drowsy_count, CLOSED_FRAMES_THRESHOLD))

            if drowsy_count >= CLOSED_FRAMES_THRESHOLD:
                label = "Drowsy"; confidence = avg
                drowsy_state["status"] = "drowsy"
            else:
                label = "Alert"; confidence = 1 - avg
                drowsy_state["status"] = "alert"
        else:
            label, confidence = "Warming up...", 0.0

        color = (0, 255, 0) if label == "Alert" else (0, 0, 255)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ----------- ROUTES -----------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    return drowsy_state["status"]

@app.route("/drowsy")
def drowsy():
    return render_template("drowsy.html")

@app.route("/rest", methods=["GET", "POST"])
def rest():
    if request.method == "POST":
        minutes = request.form.get("minutes", 5)
        try:
            minutes = int(minutes)
        except:
            minutes = 5
        return render_template("alarm.html", minutes=minutes)
    return render_template("rest.html")

# ---- GAME SELECT ----
@app.route("/game_select")
def game_select():
    return render_template("game_select.html")

@app.route("/game1")
def game1():
    return render_template("game1.html")

@app.route("/game2")
def game2():
    return render_template("game2.html")
    
@app.route("/game3")
def game3():
    return render_template("game3.html")

@app.route("/awake")
def awake():
    drowsy_state["status"] = "alert"
    return render_template("awake.html")


# -------- Start Server --------
if __name__ == "__main__":
    port = 5000
    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if s.connect_ex(('127.0.0.1', port)) == 0:
            port += random.randint(1, 40)
        else:
            break
        s.close()

    print(f"✅ Running at http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
