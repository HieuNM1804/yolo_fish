from ultralytics import YOLO
import cv2

# ===== CONFIG =====
WEIGHT_PATH = "480.pt"
VIDEO_IN = "VideoThaRubik.mp4"
VIDEO_OUT = "output_480.mp4"

IMG_SIZE = 480
CONF_THRES = 0.4
IOU_THRES = 0.5
# ==================

# 1. Load trained model
model = YOLO(WEIGHT_PATH)

# 2. Open input video
cap = cv2.VideoCapture(VIDEO_IN)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 3. Video writer (giữ nguyên resolution gốc)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

# 4. Infer + draw + save
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device="cpu",
        verbose=False
    )

    # YOLO đã vẽ sẵn bounding box
    annotated_frame = results[0].plot()

    writer.write(annotated_frame)

cap.release()
writer.release()
