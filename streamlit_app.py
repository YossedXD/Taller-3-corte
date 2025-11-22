# streamlit_app.py
"""
Versión con refresco continuo en Streamlit
y usando tus hilos, locks, semáforos y sección crítica exactamente igual.
"""

import streamlit as st
import threading
import time
import cv2
import numpy as np
import pickle
import os
import json

MODEL_PATH = "modelo_simple.pkl"
LABELS_PATH = "etl/labels.json"
SEM_LIMIT = 1
FRAME_W = 640
FRAME_H = 480

# load labels
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)
idx_to_name = {v: k for k, v in label_map.items()}

if not os.path.exists(MODEL_PATH):
    st.error(f"No se encontró el modelo: {MODEL_PATH}. Entrena primero.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    mdl = pickle.load(f)
W = mdl["W"]
b = mdl["b"]

# feature extractor
KERNELS = [
    np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32),
    np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32),
    np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
]

def extract_features_from_image(img_np):
    img = (img_np * 255.0).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
    feats = []
    for k in KERNELS:
        conv = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=k)
        conv = np.abs(conv)
        pooled = conv.reshape(32,8,32,8).mean(axis=(1,3))
        feats.append(pooled.flatten())
    feat_vec = np.concatenate(feats).astype(np.float32)
    if feat_vec.std() > 0:
        feat_vec = (feat_vec - feat_vec.mean()) / (feat_vec.std() + 1e-8)
    return feat_vec

# ==========================================
# 🔥 Cámara (thread)
# ==========================================
class CamGrabber(threading.Thread):
    def __init__(self, src=0):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.lock = threading.Lock()     # <- SECCIÓN CRÍTICA PARA FRAMES
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:         # <- SECCIÓN CRÍTICA
                    self.frame = frame.copy()
            time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        try:
            self.cap.release()
        except:
            pass

# ==========================================
# 🔥 Predictor (thread con semáforo)
# ==========================================
class PredictorThread(threading.Thread):
    def __init__(self, cam):
        super().__init__(daemon=True)
        self.cam = cam
        self.sema = threading.Semaphore(SEM_LIMIT)
        self.pred = "—"
        self.conf = 0.0
        self.running = True

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (256,256), interpolation=cv2.INTER_AREA).astype(np.float32)
        img = img / 255.0
        return img

    def run(self):
        while self.running:
            frame = self.cam.read()
            if frame is None:
                time.sleep(0.02)
                continue

            if not self.sema.acquire(blocking=False):
                time.sleep(0.01)
                continue

            try:
                img = self.preprocess(frame)
                feat = extract_features_from_image(img)
                logits = feat.dot(W.T) + b
                probs = np.exp(logits - np.max(logits))
                probs = probs / probs.sum()
                idx = int(np.argmax(probs))
                self.pred = idx_to_name[idx]
                self.conf = float(probs[idx])
            finally:
                self.sema.release()

            time.sleep(0.02)

    def stop(self):
        self.running = False

# ==========================================
# 🔥 UI
# ==========================================

st.title("🎥 Detector en vivo ")
st.write("Detección en tiempo real ")

start = st.button("Iniciar cámara")
stop_btn = st.button("Detener cámara")

if 'cam' not in st.session_state:
    st.session_state.cam = None
if 'pred' not in st.session_state:
    st.session_state.pred = None
if 'running' not in st.session_state:
    st.session_state.running = False

# Crear placeholder de video
frame_placeholder = st.empty()

# ==========================================
# 🔥 Iniciar hilos
# ==========================================
if start and not st.session_state.running:
    cam = CamGrabber(src=0)
    cam.start()
    pred_thread = PredictorThread(cam)
    pred_thread.start()
    st.session_state.cam = cam
    st.session_state.pred = pred_thread
    st.session_state.running = True
    st.success("Cámara y predictor iniciados.")

# ==========================================
# 🔥 BUCLE DE REFRESCO (video en vivo)
# ==========================================
while st.session_state.get("running", False):
    frame = st.session_state.cam.read()
    if frame is not None:
        pred_t = st.session_state.pred

        # Superponer texto
        text = f"{pred_t.pred} ({pred_t.conf:.2f})"
        cv2.putText(frame, text, (20,50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,255,0), 3, cv2.LINE_AA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB", use_column_width=True)

    time.sleep(0.01)   # evita congelar Streamlit
    st.rerun()


# ==========================================
# 🔥 Detener cámara
# ==========================================
if stop_btn and st.session_state.running:
    st.session_state.pred.stop()
    st.session_state.cam.stop()
    st.session_state.running = False
    st.success("Cámara detenida.")
    st.rerun()

