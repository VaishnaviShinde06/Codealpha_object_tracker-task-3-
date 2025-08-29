# object_tracker_app.py
import time
import tempfile
from pathlib import Path
from collections import OrderedDict


import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# --- Try importing YOLO (Ultralytics) ---
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None


def euclidean(a, b):
    ax, ay = a
    bx, by = b
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)

class CentroidTracker:
    def __init__(self, max_disappeared=20, max_distance=80):
        self.next_object_id = 1
        self.objects = OrderedDict()      # id -> (centroid, bbox, label, conf)
        self.disappeared = OrderedDict()  # id -> count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, label, conf):
        self.objects[self.next_object_id] = (centroid, bbox, label, conf)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, detections):
        """
        detections: list of dicts with keys:
          - 'bbox' = (x1, y1, x2, y2)
          - 'centroid' = (cx, cy)
          - 'label' = str
          - 'conf' = float
        """
        if len(detections) == 0:
            # mark disappeared
            to_remove = []
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    to_remove.append(object_id)
            for oid in to_remove:
                self.deregister(oid)
            return self.objects

        input_centroids = np.array([d["centroid"] for d in detections])

        if len(self.objects) == 0:
            for d in detections:
                self.register(d["centroid"], d["bbox"], d["label"], d["conf"])
            return self.objects

        # prepare lists for matching
        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[oid][0] for oid in object_ids])

        # compute pairwise distances
        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=np.float32)
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = euclidean(oc, ic)

        # greedy matching by smallest distance
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        # store updated objects temporarily
        updated_objects = {}

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            det = detections[col]
            updated_objects[object_id] = (det["centroid"], det["bbox"], det["label"], det["conf"])
            used_rows.add(row)
            used_cols.add(col)
            self.disappeared[object_id] = 0  # reset disappear counter

        # mark unmatched existing objects as disappeared
        unmatched_rows = set(range(0, D.shape[0])) - used_rows
        for row in unmatched_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
            else:
                # keep previous state if not yet deregistered
                updated_objects[object_id] = self.objects[object_id]

        # register unmatched detections as new objects
        unmatched_cols = set(range(0, D.shape[1])) - used_cols
        for col in unmatched_cols:
            det = detections[col]
            self.register(det["centroid"], det["bbox"], det["label"], det["conf"])

        # commit updates
        for oid, tup in updated_objects.items():
            self.objects[oid] = tup

        return self.objects

# ===========================
# Draw utilities
# ===========================
def draw_annotations(frame_bgr, tracked_objects, show_label=True, show_id=True):
    img = frame_bgr
    h, w = img.shape[:2]

    # Use OpenCV putText (reliable cross-platform)
    for oid, (centroid, bbox, label, conf) in tracked_objects.items():
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        cv2.rectangle(img, (x1, y1), (x2, y2), (50, 205, 50), 2)
        tag_parts = []
        if show_id:
            tag_parts.append(f"ID {oid}")
        if show_label:
            tag_parts.append(f"{label} {conf:.2f}")
        tag = " | ".join(tag_parts) if tag_parts else ""
        if tag:
            cv2.rectangle(img, (x1, y1 - 22), (x1 + 8 * len(tag) + 8, y1), (50, 205, 50), -1)
            cv2.putText(img, tag, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # draw centroid
        cx, cy = map(int, centroid)
        cv2.circle(img, (cx, cy), 3, (0, 0, 0), -1)
        cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1)
    return img

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="🎯 Object Detection & Tracking", page_icon="🎥", layout="wide")

st.markdown(
    """
    <style>
    .mini-badge {background:#eef7ee;border:1px solid #a6e4a6;color:#115522;padding:2px 8px;border-radius:999px;font-size:12px;margin-right:6px;}
    .rounded-box {border:1px solid #e6e6e6;border-radius:16px;padding:14px;background:#fff;}
    .footer-note {font-size:12px;color:#666;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎯 Real-Time Object Detection & Tracking")
st.caption("YOLOv8 + lightweight SORT-style tracker • Webcam or Video file • Live labels & IDs")

with st.sidebar:
    st.header("⚙️ Controls")
    source = st.radio("Video Source", ["Webcam", "Upload video"], index=0)

    conf_thres = st.slider("Detection Confidence", 0.1, 0.9, 0.35, 0.05)
    iou_thres = st.slider("NMS IoU Threshold", 0.2, 0.9, 0.45, 0.05)

    st.divider()
    st.subheader("Tracker Settings")
    max_disappeared = st.slider("Max disappeared frames", 5, 80, 25, 1)
    max_distance = st.slider("Max match distance (px)", 20, 200, 90, 5)

    st.divider()
    draw_labels = st.checkbox("Show class labels", True)
    draw_ids = st.checkbox("Show track IDs", True)
    show_fps = st.checkbox("Show FPS counter", True)

    st.divider()
    webcam_index = 0
    if source == "Webcam":
        webcam_index = st.number_input("Webcam index", min_value=0, value=0, step=1)
    else:
        uploaded = st.file_uploader("Upload a video file", type=["mp4", "mov", "mkv", "avi"])

    start = st.button("▶️ Start")
    stop = st.button("⏹️ Stop")

# session state for run loop
if "running" not in st.session_state:
    st.session_state.running = False

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

# Model loading guard
model_box = st.empty()
if YOLO is None:
    st.error("Ultralytics not found. Please install with: `pip install ultralytics`")
    st.stop()

try:
    model = YOLO("yolov8n.pt")
    model_box.info("Loaded model: yolov8n (small & fast).")
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}")
    st.stop()

# Video area
video_placeholder = st.empty()
stats_placeholder = st.empty()

# handle video capture
cap = None
tmp_path = None
if st.session_state.running:
    if source == "Webcam":
        cap = cv2.VideoCapture(int(webcam_index))
        if not cap.isOpened():
            st.error("Could not open webcam. Try a different index.")
            st.session_state.running = False
    else:
        if uploaded is None:
            st.warning("Please upload a video file and press Start.")
            st.session_state.running = False
        else:
            # save to a temp file so OpenCV can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tfile:
                tfile.write(uploaded.read())
                tmp_path = tfile.name
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                st.error("Could not open uploaded video.")
                st.session_state.running = False

# main loop
tracker = CentroidTracker(max_disappeared=max_disappeared, max_distance=max_distance)
prev_time = time.time()
frame_count = 0
fps = 0.0

while st.session_state.running and cap is not None and cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        st.session_state.running = False
        break

    frame_count += 1

    # YOLO inference
    results = model.predict(frame, conf=conf_thres, iou=iou_thres, verbose=False)
    detections = []
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy()
            confs = b.conf.cpu().numpy()
            clss = b.cls.cpu().numpy().astype(int)
            names = r.names

            for (x1, y1, x2, y2), conf_score, c in zip(xyxy, confs, clss):
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "centroid": (float(cx), float(cy)),
                    "label": names.get(c, str(c)),
                    "conf": float(conf_score)
                })

    # update tracker and draw
    tracked = tracker.update(detections)
    annotated = draw_annotations(frame, tracked, show_label=draw_labels, show_id=draw_ids)

    # FPS calc
    now = time.time()
    dt = now - prev_time
    if dt > 0:
        fps = 1.0 / dt
    prev_time = now

    if show_fps:
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Tracks: {len(tracked)}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)

    # display
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

    # yield to UI
    time.sleep(0.001)

# clean up
if 'cap' in locals() and cap is not None:
    cap.release()
if tmp_path:
    try:
        Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        pass

# helpful footer
st.markdown(
    """
    <div class="rounded-box">
      <span class="mini-badge">Tip</span>
      Switch to a smaller video resolution for higher FPS. Adjust the "Max match distance" if IDs flicker.
    </div>
    <p class="footer-note">Model: YOLOv8n • Tracker: centroid-based matching (SORT-style). For Deep SORT, you'd add a ReID model and a Kalman filter.</p>
    """,
    unsafe_allow_html=True
)
