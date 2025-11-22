# train_model.py
"""
Entrenador "CNN de juguete" en NumPy + OpenCV.
- Usa hilos para lectura de .npy (producer)
- Semáforo limita lecturas concurrentes
- Lock protege indices / sección crítica
- Extrae features aplicando filtros conv. con cv2.filter2D
- Pooling por downsample simple
- Clasificador softmax lineal entrenado con SGD
- Guarda modelo en modelo_simple.pkl
"""

import os
import json
import time
import threading
from queue import Queue
import pickle

import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# -----------------------
# CONFIG
# -----------------------
DATASET_DIR = "etl/data/processed"
LABELS_PATH = "etl/labels.json"
MODEL_OUT = "modelo_simple.pkl"

NUM_WORKERS = 6            # hilos lectores
SEMAPHORE_LIMIT = 3        # cuántas lecturas a disco a la vez
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.01
QUEUE_MAX = 256            # máximo batches en cola
DEVICE = "cpu"

# -----------------------
# SAMPLE UPLOADED FILE (dev note)
# -----------------------
SAMPLE_FILE = "/mnt/data/a1271338-54f0-4356-86ba-0d6f957c4f86.png"

# -----------------------
# Cargar labels y rutas
# -----------------------
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

print("Labels:", label_map)
idx_to_name = {v:k for k, v in label_map.items()}

def list_files_and_labels(dataset_dir, label_map):
    files = []
    labels = []
    for clase in os.listdir(dataset_dir):
        cpath = os.path.join(dataset_dir, clase)
        if not os.path.isdir(cpath):
            continue
        lab = label_map.get(clase)
        if lab is None:
            continue
        for fn in os.listdir(cpath):
            if fn.endswith(".npy"):
                files.append(os.path.join(cpath, fn))
                labels.append(lab)
    return files, labels

files_all, labels_all = list_files_and_labels(DATASET_DIR, label_map)
print("Total .npy found:", len(files_all))

# split
train_files, test_files, train_labels, test_labels = train_test_split(
    files_all, labels_all, test_size=0.20, random_state=42, shuffle=True
)

# -----------------------
# Util: feature extractor (conv filters + downsample pooling)
# -----------------------
# We'll use a set of fixed kernels (edge detectors) to emulate small conv filters.
KERNELS = [
    np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32),  # sobel x
    np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32),  # sobel y
    np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)  # sharpen-ish
]

def extract_features_from_image(img_np):
    """
    img_np: 2D array (256x256) float32 in [0..1]
    returns: 1D feature vector (concatenated pooled feature maps)
    """
    # ensure float32
    img = (img_np * 255.0).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
    # work in grayscale (ETL produced grayscale images)
    feats = []
    for k in KERNELS:
        conv = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=k)
        # absolute + normalize
        conv = np.abs(conv)
        # downsample / pooling: reduce 256->32 with simple block mean (factor 8)
        pooled = conv.reshape(32,8,32,8).mean(axis=(1,3))  # (32,32)
        feats.append(pooled.flatten())
    feat_vec = np.concatenate(feats)  # 3 * (32*32) = 3072 dim
    # normalize features
    feat_vec = feat_vec.astype(np.float32)
    if feat_vec.std() > 0:
        feat_vec = (feat_vec - feat_vec.mean()) / (feat_vec.std() + 1e-8)
    return feat_vec  # shape (3072,)

# -----------------------
# Threaded producer: lee archivos .npy y pone (feature,label) en cola
# -----------------------
class Producer:
    def __init__(self, files, labels, queue, sem_limit=SEMAPHORE_LIMIT):
        self.files = files
        self.labels = labels
        self.n = len(files)
        self.index = 0
        self.lock = threading.Lock()
        self.queue = queue
        self.sema = threading.Semaphore(sem_limit)
        self.threads = []
        self.stop_event = threading.Event()

    def _next_pair(self):
        with self.lock:
            if self.index >= self.n:
                return None, None
            f = self.files[self.index]
            l = self.labels[self.index]
            self.index += 1
            return f, l

    def _worker(self):
        while not self.stop_event.is_set():
            pair = self._next_pair()
            if pair[0] is None:
                return
            path, lab = pair
            # limit IO concurrency
            self.sema.acquire()
            try:
                arr = np.load(path).astype(np.float32)  # already normalized 0..1
            except Exception as e:
                print("Error loading", path, e)
                arr = None
            finally:
                self.sema.release()

            if arr is None:
                continue

            # feature extraction (CPU-heavy) - not in critical section
            feat = extract_features_from_image(arr)

            # sección crítica: poner en cola
            self.queue.put((feat, lab))

    def start(self, num_workers=NUM_WORKERS):
        self.stop_event.clear()
        for _ in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self):
        self.stop_event.set()
        # wait threads to finish briefly
        for t in self.threads:
            t.join(timeout=1)

# -----------------------
# Simple softmax linear classifier (weights, bias) with SGD
# -----------------------
class SoftmaxClassifier:
    def __init__(self, input_dim, num_classes):
        self.W = np.random.randn(num_classes, input_dim).astype(np.float32) * 0.01
        self.b = np.zeros((num_classes,), dtype=np.float32)

    def predict_logits(self, X):
        # X shape (B, D)
        return X.dot(self.W.T) + self.b  # (B, C)

    def predict_proba(self, X):
        logits = self.predict_logits(X)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        return p

    def predict(self, X):
        p = self.predict_proba(X)
        return p.argmax(axis=1)

    def train_sgd(self, X_batch, y_batch, lr=0.01):
        # X_batch (B,D), y_batch (B,)
        B = X_batch.shape[0]
        logits = self.predict_logits(X_batch)  # (B,C)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        # one-hot
        onehot = np.zeros_like(probs)
        onehot[np.arange(B), y_batch] = 1.0
        # gradients
        grad_logits = (probs - onehot) / B  # (B,C)
        dW = grad_logits.T.dot(X_batch)     # (C,D)
        db = grad_logits.sum(axis=0)        # (C,)
        # update
        self.W -= lr * dW
        self.b -= lr * db
        # cross-entropy loss
        log_lik = -np.log(probs[np.arange(B), y_batch] + 1e-12)
        loss = log_lik.mean()
        acc = (probs.argmax(axis=1) == y_batch).mean()
        return loss, acc

# -----------------------
# Training loop (consumer reads from queue)
# -----------------------
def run_training():
    # queue holds single samples; consumer will build batches
    sample_queue = Queue(maxsize=QUEUE_MAX)
    producer = Producer(train_files, train_labels, sample_queue)
    print("Starting producer threads...")
    producer.start()

    # initialize model
    # sample one element to infer dimension (wait for producer to fill)
    print("Waiting for first sample to infer feature dim...")
    while sample_queue.empty():
        time.sleep(0.1)
    feat0, _ = sample_queue.get()
    D = feat0.shape[0]
    C = len(label_map)
    model = SoftmaxClassifier(input_dim=D, num_classes=C)
    print(f"Model init: input_dim={D}, num_classes={C}")

    # put it back
    sample_queue.put((feat0, _))

    # training
    best_val = -1.0

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        # training pass: consume train set
        train_loss_accum = 0.0
        train_acc_accum = 0.0
        train_batches = 0

        # we will sample until we've consumed all train samples
        consumed = 0
        total_train = len(train_files)

        while consumed < total_train:
            # build batch
            X_batch = []
            y_batch = []
            for _ in range(BATCH_SIZE):
                try:
                    feat, lab = sample_queue.get(timeout=5)
                except:
                    break
                X_batch.append(feat)
                y_batch.append(lab)
                consumed += 1
            if len(X_batch) == 0:
                break
            Xb = np.stack(X_batch).astype(np.float32)
            yb = np.array(y_batch, dtype=np.int64)
            loss, acc = model.train_sgd(Xb, yb, lr=LEARNING_RATE)
            train_loss_accum += loss
            train_acc_accum += acc
            train_batches += 1

        # validate on test set (single-threaded, deterministic)
        # build test features directly (no threads)
        X_test_feats = []
        y_test_vals = []
        for pth in test_files:
            arr = np.load(pth).astype(np.float32)
            feat = extract_features_from_image(arr)
            X_test_feats.append(feat)
            # label from mapping test_files->test_labels
            # find index
        # map test_files -> labels by the earlier split
        X_test_feats = np.stack(X_test_feats).astype(np.float32)
        y_test_vals = np.array(test_labels, dtype=np.int64)

        probs = model.predict_proba(X_test_feats)
        val_preds = probs.argmax(axis=1)
        val_acc = (val_preds == y_test_vals).mean() * 100.0

        t1 = time.time()

        # metrics
        avg_train_loss = train_loss_accum / max(1, train_batches)
        avg_train_acc = (train_acc_accum / max(1, train_batches)) * 100.0

        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {avg_train_loss:.4f} - train_acc: {avg_train_acc:.2f}% - val_acc: {val_acc:.2f}% - time: {t1-t0:.1f}s")

        # guardar mejor
        if val_acc > best_val:
            best_val = val_acc
            with open(MODEL_OUT, "wb") as f:
                pickle.dump({"W": model.W, "b": model.b, "D": D, "C": C}, f)
            print("Saved model:", MODEL_OUT)

        # reset producer for next epoch (rewind)
        producer.stop()
        sample_queue = Queue(maxsize=QUEUE_MAX)
        producer = Producer(train_files, train_labels, sample_queue)
        producer.start()

    # fin
    producer.stop()
    print("Training finished. Best val acc: %.2f%%" % best_val)

if __name__ == "__main__":
    run_training()
