import os
import base64
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from model_utils import load_gesture_model, preprocess_frame
from gesture_actions import perform_action

# SETTINGS
MODEL_PATH = os.environ.get("MODEL_PATH", "./model/gesture_model.h5")
ENABLE_SERVER_ACTIONS = os.environ.get("ENABLE_SERVER_ACTIONS", "false").lower() == "true"
CLASS_LABELS_PATH = "./model/labels.txt"   # optional: one label per line

app = Flask(__name__, static_folder="static", template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")  # uses eventlet/gevent if installed

# Load model once
model = load_gesture_model(MODEL_PATH)

# Try to load labels if available
if os.path.exists(CLASS_LABELS_PATH):
    with open(CLASS_LABELS_PATH, "r") as f:
        class_labels = [l.strip() for l in f if l.strip()]
else:
    # fallback labels — replace with your real classes
    class_labels = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
        'del', 'nothing', 'space'
    ]


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("frame")
def handle_frame(data_url):
    try:
        # Accept either "data:image/jpeg;base64,..." or raw base64 string
        if "," in data_url:
            b64 = data_url.split(",", 1)[1]
        else:
            b64 = data_url
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # preprocess and predict
        img = preprocess_frame(frame)  # returns batch (1, H, W, C)
        preds = model.predict(img, verbose=0)[0]

        # take top 3 predictions
        top_idxs = preds.argsort()[-3:][::-1]
        results = [
            {"label": class_labels[i], "confidence": float(preds[i])}
            for i in top_idxs
        ]

        # Optionally perform local server-side action
        if ENABLE_SERVER_ACTIONS:
            perform_action(results[0]["label"])  # top prediction only

        # send top predictions back
        emit("prediction", {"top_predictions": results})

    except Exception as e:
        emit("prediction", {"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # For development this will detect eventlet/gevent if installed per Flask-SocketIO docs.
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
