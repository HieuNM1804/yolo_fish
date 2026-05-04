from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
from collections import deque
from math import sqrt
from scipy.optimize import linear_sum_assignment

VIDEO_IN = "fishing/8.mp4"
WEIGHT_PATH = "480.pt"
VIDEO_OUT = "track_480.mp4"

# YOLO config
IMG_SIZE = 480
CONF_THRES = 0.75
IOU_THRES = 0.5

#CONFIG 2 

SCALE = 0.4             
DETECT_EVERY = 1
MAX_AGE = 1
MAX_AREA = 10000 # Max area for detection
EXIT_MARGIN = 50
RIGHT_EXIT_RATIO = 0.99
LEFT_DET_RATIO = 0.05
MIN_TRACK_AGE = 2
RECENT_EXIT_TTL = 15
RECENT_EXIT_DIST = 20
VERBOSE = os.getenv('VERBOSE', 'False') == 'True'
SHOW_VIS = os.getenv('SHOW_VIS', 'True') == 'True'
SAVE_OUTPUT = os.getenv('SAVE_OUTPUT', 'True') == 'True'

FAST_LOST_MISSED = 1
TRACK_SPEED_THRESHOLD = 1.0
FAST_LOST_RIGHT_MARGIN = 1000
INITIAL_VX = 130.0
INITIAL_VY = 30.0


class KalmanSimple:
    def __init__(self, cx, cy, dt=1.0):
        self.dt = float(dt)
        self.x = np.array([cx, cy, INITIAL_VX, INITIAL_VY], dtype=float)
        self.P = np.eye(4) * 50.0
        self.F = np.array([[1,0,self.dt,0],
                        [0,1,0,self.dt],
                        [0,0,1,0],
                        [0,0,0,1]], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.Q = np.eye(4) * 5.0
        self.R = np.eye(2) * 20.0

    def predict(self):
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.x.copy()

    def update(self, meas):
        z = np.array([meas[0], meas[1]], dtype=float)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        y = z - self.H.dot(self.x)
        self.x = self.x + K.dot(y)
        self.P = (np.eye(4) - K.dot(self.H)).dot(self.P)

    def state(self):
        return self.x.copy()

def assign(cost):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    r,c = linear_sum_assignment(cost)
    matches = []; unmatched_t = list(range(cost.shape[0])); unmatched_d = list(range(cost.shape[1]))
    for i,j in zip(r,c):
        if cost[i,j] > 1e5: continue
        matches.append((i,j))
        if i in unmatched_t: unmatched_t.remove(i)
        if j in unmatched_d: unmatched_d.remove(j)
    return matches, unmatched_t, unmatched_d

def find_detections_yolo(frame, model, DET_LEFT_X, DET_RIGHT_X):
    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device="cpu",
        verbose=False
    )
    
    dets = []
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box
            x, y = int(x1), int(y1)
            wc, hc = int(x2 - x1), int(y2 - y1)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if cx < DET_LEFT_X or cx > DET_RIGHT_X:
                continue
            
            area = wc * hc

            if area > MAX_AREA:
                continue

            dets.append({
                'bbox': (x, y, wc, hc),
                'centroid': (cx, cy),
                'tlbr': (int(x1), int(y1), int(x2), int(y2)),
                'area': area
            })

    return dets

if WEIGHT_PATH.endswith('.pt'):
    ov_path = os.path.splitext(WEIGHT_PATH)[0] + "_openvino_model"
    # NOTE: Delete the folder '480_openvino_model' if you want to re-export with INT8 quantization
    if os.path.exists(ov_path):
        print(f"[INFO] Using existing OpenVINO model: {ov_path}")
        WEIGHT_PATH = ov_path
    else:
        print(f"[INFO] Exporting {WEIGHT_PATH} to OpenVINO (INT8 Quantization)...")
        try:
            model_pt = YOLO(WEIGHT_PATH)
            # Use data='dataset/data.yaml' for calibration during INT8 export
            model_pt.export(format='openvino', int8=True, data='dataset/data.yaml')
            WEIGHT_PATH = ov_path
            print(f"[INFO] Export success. New weights: {WEIGHT_PATH}")
        except Exception as e:
            print(f"[WARN] INT8 Export failed: {e}. Falling back to standard OpenVINO export...")
            try:
                 model_pt.export(format='openvino')
                 WEIGHT_PATH = ov_path
                 print(f"[INFO] Standard OpenVINO export success. New weights: {WEIGHT_PATH}")
            except Exception as e2:
                 print(f"[WARN] Standard Export also failed: {e2}. Using original .pt model.")

model = YOLO(WEIGHT_PATH)
cap = cv2.VideoCapture(VIDEO_IN)
ret, f0 = cap.read()


h0, w0 = f0.shape[:2]
w = int(w0 * SCALE); h = int(h0 * SCALE)
DET_LEFT_X = int(w * LEFT_DET_RATIO)
EXIT_X = int(w * RIGHT_EXIT_RATIO)
DET_RIGHT_X = EXIT_X

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0 or np.isnan(fps):
    fps = 30.0

writer = None
if SAVE_OUTPUT:
    fourcc_try = ['mp4v', 'XVID', 'avc1']
    for codec in fourcc_try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(VIDEO_OUT, fourcc, float(fps), (w, h), True)
        if writer is not None and writer.isOpened():
            print(f"[INFO] Writing output to {VIDEO_OUT} with codec {codec} at {fps:.2f} FPS")
            break
        else:
            writer = None

tracks = {}
next_id = 1
frame_idx = 0
total_count = 0
recent_exits = []

start = time.time()
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame_raw = cap.read()
    # if not ret:
    #     # Loop video infinitely
    #     print("[INFO] Video ended, rewinding to start for infinite loop...")
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #     ret, frame_raw = cap.read()
    if not ret:
        break
    frame_idx += 1
    frame = cv2.resize(frame_raw, (w, h)) 

    dets = []
    if frame_idx % DETECT_EVERY == 0:
        dets = find_detections_yolo(frame, model, DET_LEFT_X, DET_RIGHT_X)

    pred_boxes = []; track_ids = []; track_preds = []
    for tid, t in list(tracks.items()):
        st = t['kf'].predict()
        pred_cx = float(st[0]); pred_cy = float(st[1])
        pred_boxes.append([pred_cx - t['bbox'][2]/2.0, pred_cy - t['bbox'][3]/2.0,
                        pred_cx + t['bbox'][2]/2.0, pred_cy + t['bbox'][3]/2.0])
        track_ids.append(tid)
        track_preds.append((pred_cx, pred_cy, st))

    if len(tracks) > 0 and len(dets) > 0:
        N = len(track_ids); M = len(dets)
        cost = np.ones((N,M), dtype=float) * 1e6
        for ii in range(N):
            pred_cx, pred_cy, st = track_preds[ii]
            
            # Constraint: Movement is strictly Left -> Right
            # Do not match if detection is to the left of the last known position
            tid = track_ids[ii]
            last_cx = tracks[tid]['centroid'][0] # Last known real position

            vx = float(st[2]); vy = float(st[3])
            speed = sqrt(vx*vx + vy*vy)
            gating = max(100.0, 100.0 + 3.0 * speed)
            for j in range(M):
                dcx, dcy = dets[j]['centroid']

                # "không được gán id cho một object đằng trước nó" -> dcx must be >= last_cx
                # Allow a small jitter margin (e.g. 10 pixels)
                if dcx < (last_cx - 10): 
                    continue

                d = sqrt((pred_cx-dcx)**2 + (pred_cy-dcy)**2)
                if d <= gating:
                    cost[ii,j] = d
        matches_raw, unmatched_t_raw, unmatched_d_raw = assign(cost)

        matches = []; matched_tids = set(); matched_dinds = set()
        for ti,dj in matches_raw:
            if ti < 0 or dj < 0: continue
            if cost[ti,dj] > 1e5: continue
            matches.append((ti,dj)); matched_tids.add(ti); matched_dinds.add(dj)

        unmatched_t = [i for i in range(N) if i not in matched_tids]
        unmatched_d = [j for j in range(M) if j not in matched_dinds]

        # Update matched
        for ti, dj in matches:
            tid = track_ids[ti]
            if tid not in tracks: continue
            det = dets[dj]
            bx,by,bw_box,bh_box = det['bbox']; cx,cy = det['centroid']
            tracks[tid]['kf'].update([cx,cy])
            tracks[tid]['bbox'] = (bx,by,bw_box,bh_box)
            tracks[tid]['centroid'] = (cx,cy)
            tracks[tid]['missed'] = 0
            tracks[tid]['history'].append((cx,cy))
            tracks[tid]['area'] = det.get('area', bw_box*bh_box)
            if len(tracks[tid]['history']) >= 2:
                (x1,y1) = tracks[tid]['history'][-2]; (x2,y2) = tracks[tid]['history'][-1]
                vx = x2 - x1; vy = y2 - y1
                tracks[tid]['kf'].x[2] = 0.6 * tracks[tid]['kf'].x[2] + 0.4 * vx
                tracks[tid]['kf'].x[3] = 0.6 * tracks[tid]['kf'].x[3] + 0.4 * vy

        for idx in unmatched_t:
            if idx < 0 or idx >= len(track_ids): continue
            tid = track_ids[idx]
            if tid in tracks:
                tracks[tid]['missed'] += 1

        to_fast_lost = []
        for tid, t in list(tracks.items()):
            missed = t.get('missed',0)
            age = t.get('age',1)
            st = t['kf'].state()
            pred_cx = float(st[0]); pred_cy = float(st[1])
            vx = float(st[2]); vy = float(st[3])
            bw_box = t['bbox'][2] if 'bbox' in t and t.get('bbox') is not None else 20
            right_edge_pred = pred_cx + bw_box/2.0
            
            # Logic: If track is missed
            if missed > 0:
                # Condition 1: Fast Lost at Exit (Right side)
                is_fast_lost = (missed >= FAST_LOST_MISSED and age >= MIN_TRACK_AGE and 
                              vx >= TRACK_SPEED_THRESHOLD and 
                              right_edge_pred >= (EXIT_X - FAST_LOST_RIGHT_MARGIN))
                
                # Condition 2: Lost on Left side (missed) - count immediately
                # "vì cá luôn bơi trái sang phải... loại bỏ luôn id đó và tăng cnt"
                # Ensure age is sufficient to avoid noise
                is_lost_on_left = (pred_cx < EXIT_X) and (age >= MIN_TRACK_AGE)

                if is_fast_lost or is_lost_on_left:
                    cx_hist = t['centroid'][0] if t.get('centroid') is not None else pred_cx
                    cy_hist = t['centroid'][1] if t.get('centroid') is not None else pred_cy

                    duplicate = False
                    for re in recent_exits:
                        if frame_idx - re['frame'] > RECENT_EXIT_TTL:
                            continue
                        dist = sqrt((cx_hist - re['cx'])**2 + (cy_hist - re['cy'])**2)
                        if dist <= RECENT_EXIT_DIST:
                            duplicate = True
                            break
                    if duplicate:
                        if VERBOSE:
                            print(f"[FAST_LOST_SKIP_DUPLICATE] tid={tid} skip")
                        continue
                    
                    to_fast_lost.append((tid, cx_hist, cy_hist, pred_cx))
        for (tid, cx_hist, cy_hist, pred_cx) in to_fast_lost:
            t = tracks.get(tid, None)
            if t is None:
                continue
            recent_exits.append({'cx': float(cx_hist), 'cy': float(cy_hist), 'frame': frame_idx})
            bx,by,bw_box,bh_box = t['bbox']
            area_rect = t.get('area', bw_box*bh_box)
            total_count += 1
            try:
                del tracks[tid]
            except Exception:
                pass

        for j in unmatched_d:
            if j < 0 or j >= len(dets): continue
            det = dets[j]
            bx,by,bw_box,bh_box = det['bbox']; cx,cy = det['centroid']
            right_edge_det = bx + bw_box
            if right_edge_det >= (EXIT_X - EXIT_MARGIN):
                if VERBOSE:
                    print(f"[SKIP_CREATE] det at/right of EXIT_X, skip creation cx={cx} right_edge={right_edge_det:.1f}")
                continue
            skip_due_recent_exit = False
            for re in recent_exits:
                if frame_idx - re['frame'] > RECENT_EXIT_TTL:
                    continue
                dist = sqrt((cx - re['cx'])**2 + (cy - re['cy'])**2)
                if dist <= RECENT_EXIT_DIST:
                    skip_due_recent_exit = True
                    if VERBOSE:
                        print(f"[SKIP_CREATE_NEAR_RECENT_EXIT] dist={dist:.1f} cx={cx} cy={cy}")
                    break
            if skip_due_recent_exit:
                continue
            kf = KalmanSimple(cx, cy)
            tracks[next_id] = {'kf': kf, 'bbox':(bx,by,bw_box,bh_box), 'centroid':(cx,cy),
                            'missed':0, 'prev_cx':cx, 'history': deque([(cx,cy)], maxlen=5), 
                            'age':1, 'area': det.get('area', bw_box*bh_box)}
            if VERBOSE:
                print(f"[NEW] id={next_id} bbox={(bx,by,bw_box,bh_box)} area={tracks[next_id]['area']}")
            next_id += 1

    else:
        if len(dets) > 0:
            for det in dets:
                bx,by,bw_box,bh_box = det['bbox']; cx,cy = det['centroid']
                right_edge_det = bx + bw_box
                if right_edge_det >= (EXIT_X - EXIT_MARGIN):
                    if VERBOSE:
                        print(f"[SKIP_CREATE_TOPLEVEL] det at/right of EXIT_X, skip creation cx={cx} right_edge={right_edge_det:.1f}")
                    continue
                skip_due_recent_exit = False
                for re in recent_exits:
                    if frame_idx - re['frame'] > RECENT_EXIT_TTL:
                        continue
                    dist = sqrt((cx - re['cx'])**2 + (cy - re['cy'])**2)
                    if dist <= RECENT_EXIT_DIST:
                        skip_due_recent_exit = True
                        if VERBOSE:
                            print(f"[SKIP_CREATE_NEAR_RECENT_EXIT_TOP] dist={dist:.1f} cx={cx} cy={cy}")
                        break
                if skip_due_recent_exit:
                    continue
                kf = KalmanSimple(cx, cy)
                tracks[next_id] = {'kf': kf, 'bbox':(bx,by,bw_box,bh_box), 'centroid':(cx,cy),
                                'missed':0, 'prev_cx':cx, 'history': deque([(cx,cy)], maxlen=5), 
                                'age':1, 'area': det.get('area', bw_box*bh_box)}
                if VERBOSE:
                    print(f"[NEW] id={next_id} bbox={(bx,by,bw_box,bh_box)} area={tracks[next_id]['area']}")
                next_id += 1

    for tid, t in list(tracks.items()):
        bx,by,bw_box,bh_box = t['bbox']
        right_edge = float(bx + bw_box)
        if right_edge >= (EXIT_X - EXIT_MARGIN) and t.get('age',0) >= MIN_TRACK_AGE:
            cx_hist = t['centroid'][0] if t.get('centroid') is not None else (t['history'][-1][0] if len(t['history'])>0 else None)
            cy_hist = t['centroid'][1] if t.get('centroid') is not None else (t['history'][-1][1] if len(t['history'])>0 else None)
            if cx_hist is None:
                cx_hist = (bx + bx + bw_box) / 2.0
                cy_hist = (by + by + bh_box) / 2.0
            recent_exits.append({'cx': float(cx_hist), 'cy': float(cy_hist), 'frame': frame_idx})
            st = t['kf'].state()
            pred_cx = float(st[0])
            prev_cx = t.get('prev_cx', None)
            area_rect = t.get('area', bw_box*bh_box)
            total_count += 1
            t['exiting'] = True
            if VERBOSE:
                print(f"[EXIT_IMMEDIATE] frame={frame_idx} id={tid} right_edge={right_edge:.1f} EXIT_X={EXIT_X} TOTAL={total_count}")

    to_del = []
    for tid, t in list(tracks.items()):
        if t['missed'] > MAX_AGE:
            to_del.append(tid)
        if t.get('exiting', False):
            to_del.append(tid)
    for tid in to_del:
        if VERBOSE and tid in tracks:
            print(f"[DEL] id={tid} missed={tracks[tid]['missed']}")
        if tid in tracks:
            del tracks[tid]

    recent_exits = [re for re in recent_exits if frame_idx - re['frame'] <= RECENT_EXIT_TTL]
    for tid, t in tracks.items():
        if len(t['history'])>0:
            t['prev_cx'] = t['history'][-1][0]
        t['age'] = t.get('age', 1) + 1

    # Drawing is done if visualize is ON OR if we need to save the output video
    vis = None
    if SHOW_VIS or SAVE_OUTPUT:
        vis = frame.copy()
        cv2.line(vis, (DET_LEFT_X,0), (DET_LEFT_X,h), (0,200,200), 2)
        cv2.line(vis, (EXIT_X,0), (EXIT_X,h), (0,0,255), 2)

        if dets is not None:
            for d in dets:
                bx,by,bw_box,bh_box = d['bbox']
                cv2.rectangle(vis, (int(bx),int(by)), (int(bx+bw_box), int(by+bh_box)), (0,255,0), 1)

        for tid, t in tracks.items():
            bx,by,bw_box,bh_box = t['bbox']
            cx,cy = t['centroid']

            if t.get('exiting', False) or (bx + bw_box) >= (EXIT_X - EXIT_MARGIN):
                color = (0,0,255)  
            else:
                color = (0,255,255) 
            
            cv2.rectangle(vis, (int(bx),int(by)), (int(bx+bw_box), int(by+bh_box)), color, 2)
            cv2.circle(vis, (int(cx), int(cy)), 3, color, -1)
            cv2.putText(vis, f"{tid}", (int(cx)+5,int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1)
        
        cv2.putText(vis, f"Count: {total_count}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0),2)

    if SHOW_VIS and vis is not None:
        cv2.imshow('vis', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if SAVE_OUTPUT and writer is not None and vis is not None:
        try:
            writer.write(vis)
        except Exception as e:
            if VERBOSE:
                print("[WARN] writer.write failed:", e)
            writer = None


end = time.time()
cap.release()
if writer is not None:
    writer.release()
    if VERBOSE:
        print(f"[INFO] Saved output to {VIDEO_OUT}")
if SHOW_VIS:
    cv2.destroyAllWindows()

print(f"Finished {os.path.basename(VIDEO_IN)}. Total counted (exit right): {total_count}. Elapsed(s): {end-start:.2f}")
print("Done.")
