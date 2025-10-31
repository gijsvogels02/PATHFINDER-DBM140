# grid_counter4.py
# Fast people grid counter with YOLO + homography (center point)
# Exports a live Python file (script1.py) containing the current people grid

import os, time, json
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import torch

# ---- limit PyTorch threads on Pi ----
try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# ========= SETTINGS =========
MODEL         = "yolov8n.pt"   # try "yolov8s.pt" for better recall
CONF          = 0.20
IOU           = 0.75
IMGSZ         = 640
PROCESS_EVERY = 5
MAX_DET       = 60

CAP_W, CAP_H     = 640, 360
LORES_W, LORES_H = 416, 234
GRID_ROWS, GRID_COLS = 10, 10
EMA_ALPHA        = 0.5
BOX_PERSIST      = 5

MIN_AREA = 25
MAX_AREA = 6000

BOX_X_OFFSET = 20
BOX_Y_OFFSET = 0

DEFAULT_CALIB_SRC_W = 1280
DEFAULT_CALIB_SRC_H = 720
# =============================

def cell_of_point_map(x, y, rows, cols, wpx, hpx):
    cw, ch = wpx / cols, hpx / rows
    c = int(np.clip(x // cw, 0, cols - 1))
    r = int(np.clip(y // ch, 0, rows - 1))
    return r, c


def render_birdeye_map(rows, cols, counts, wpx, hpx, dots=None):
    img = np.zeros((hpx, wpx, 3), dtype=np.uint8)
    for r in range(1, rows):
        y = int(r * hpx / rows)
        cv2.line(img, (0, y), (wpx - 1, y), (80, 80, 80), 1)
    for c in range(1, cols):
        x = int(c * wpx / cols)
        cv2.line(img, (x, 0), (x, hpx - 1), (80, 80, 80), 1)
    for r in range(rows):
        for c in range(cols):
            cx = int((c + 0.5) * wpx / cols)
            cy = int((r + 0.5) * hpx / rows)
            cv2.putText(img, str(int(counts[r][c])), (cx - 8, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    if dots:
        for (wx, wy) in dots:
            cv2.circle(img, (int(wx), int(wy)), 3, (0, 255, 0), -1)
    return img


def main():
    global BOX_X_OFFSET, BOX_Y_OFFSET

    # --- Load homography + meta ---
    try:
        H = np.load("H.npy")
        with open("homography_meta.json", "r") as f:
            meta = json.load(f)
        OUT_WPX = int(meta["OUT_WPX"])
        OUT_HPX = int(meta["OUT_HPX"])
        CALIB_SRC_W = int(meta.get("SRC_W", DEFAULT_CALIB_SRC_W))
        CALIB_SRC_H = int(meta.get("SRC_H", DEFAULT_CALIB_SRC_H))
    except Exception as e:
        raise SystemExit("Missing H.npy / homography_meta.json. "
                         "Re-run calibration; defaults assume 1280x720 source.") from e

    # --- Build polygon mask (in LORES) for the mat ---
    Hinv = np.linalg.inv(H)
    map_corners = np.array([
        [0, 0],
        [OUT_WPX-1, 0],
        [OUT_WPX-1, OUT_HPX-1],
        [0, OUT_HPX-1]
    ], dtype=np.float32).reshape(1,4,2)
    src_poly = cv2.perspectiveTransform(map_corners, Hinv)[0]
    sxL, syL = LORES_W / float(CALIB_SRC_W), LORES_H / float(CALIB_SRC_H)
    lores_poly = (src_poly * np.array([sxL, syL], dtype=np.float32)).astype(np.int32)
    roi_mask = np.zeros((LORES_H, LORES_W), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [lores_poly], 255)

    # --- Camera setup ---
    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": (CAP_W, CAP_H), "format": "RGB888"},
        lores={"size": (LORES_W, LORES_H), "format": "YUV420"},
        buffer_count=4
    )
    cam.configure(config)
    cam.start()
    time.sleep(0.2)
    try:
        cam.set_controls({"Sharpness": 1.15, "Contrast": 1.05})
    except Exception:
        pass

    # --- YOLO model + warmup ---
    model = YOLO(MODEL)
    _ = model.predict(
        source=np.zeros((LORES_H, LORES_W, 3), dtype=np.uint8),
        imgsz=IMGSZ, conf=CONF, iou=IOU, classes=[0], max_det=1, verbose=False
    )

    frames = 0
    ema_counts = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    last_boxes = np.empty((0, 4))
    box_age = 0
    last_infer_ms = 0.0
    t0 = time.time()

    while True:
        frame = cam.capture_array()
        lores = cam.capture_array("lores")
        lores = cv2.cvtColor(lores, cv2.COLOR_YUV420p2RGB)
        hL, wL = lores.shape[:2]

        counts_now = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
        warped_dots = []

        # ---- Run YOLO every Nth frame ----
        if frames % PROCESS_EVERY == 0:
            t_inf = time.time()
            res = model.predict(
                source=lores, imgsz=IMGSZ, conf=CONF, iou=IOU,
                classes=[0], max_det=MAX_DET, agnostic_nms=True,
                augment=True, verbose=False
            )[0]
            last_boxes = res.boxes.xyxy.cpu().numpy() if (res.boxes is not None and len(res.boxes) > 0) else np.empty((0, 4))
            box_age = 0 if last_boxes.size else box_age
            last_infer_ms = (time.time() - t_inf) * 1000.0
        else:
            box_age += 1

        # ---- Draw boxes + COUNT ----
        if len(last_boxes) > 0 and box_age < BOX_PERSIST:
            sx, sy = CAP_W / wL, CAP_H / hL
            scale_x = CALIB_SRC_W / float(wL)
            scale_y = CALIB_SRC_H / float(hL)

            # per-cell duplicate guard
            per_cell_cap = 4
            per_cell_counts = {}

            for (x1, y1, x2, y2) in last_boxes:
                # area/shape gate
                w = float(x2 - x1); h = float(y2 - y1)
                area = w * h
                if area < MIN_AREA or area > MAX_AREA:
                    continue
                ar = h / (w + 1e-6)
                if ar < 1.2 or ar > 5.0:
                    continue

                fx, fy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                if roi_mask[int(np.clip(fy,0,LORES_H-1)), int(np.clip(fx,0,LORES_W-1))] == 0:
                    continue

                # Draw box on preview
                X1 = int(x1 * sx) + BOX_X_OFFSET
                Y1 = int(y1 * sy) + BOX_Y_OFFSET
                X2 = int(x2 * sx) + BOX_X_OFFSET
                Y2 = int(y2 * sy) + BOX_Y_OFFSET
                cv2.rectangle(frame, (X1, Y1), (X2, Y2), (0, 255, 0), 2)

                # Homography warp
                fx_cal = fx * scale_x
                fy_cal = fy * scale_y
                pt = np.array([[[fx_cal, fy_cal]]], dtype=np.float32)
                warped = cv2.perspectiveTransform(pt, H)[0, 0]
                wx, wy = float(warped[0]), float(warped[1])

                if 0 <= wx < OUT_WPX and 0 <= wy < OUT_HPX:
                    r, c = cell_of_point_map(wx, wy, GRID_ROWS, GRID_COLS, OUT_WPX, OUT_HPX)

                    key = (r, c)
                    if per_cell_counts.get(key, 0) < per_cell_cap:
                        counts_now[r, c] += 1
                        per_cell_counts[key] = per_cell_counts.get(key, 0) + 1
                        warped_dots.append((wx, wy))

        # ---- Smooth counts ----
        ema_counts = (1 - EMA_ALPHA) * ema_counts + EMA_ALPHA * counts_now
        counts_disp = np.rint(ema_counts).astype(int)

        # ---- Export live grid ----
        try:
            with open("script1.py", "w") as f:
                f.write("def func():\n")
                f.write("    global matrix\n")
                f.write("    matrix = [\n")
                for row in counts_disp.tolist():
                    f.write(f"    {row},\n")
                f.write("]\nfunc()\n")
        except Exception as e:
            print("⚠️ Could not write script1.py:", e)

        # ---- Display ----
        cv2.imshow("Camera view (q to quit)", frame)
        map_img = render_birdeye_map(GRID_ROWS, GRID_COLS, counts_disp, OUT_WPX, OUT_HPX, dots=warped_dots)
        cv2.imshow("Bird's-eye occupancy", map_img)

        # ---- FPS + keyboard ----
        frames += 1
        if frames % 40 == 0:
            fps = 40 / (time.time() - t0); t0 = time.time()
            if last_boxes.size:
                areas = [float((x2-x1)*(y2-y1)) for (x1,y1,x2,y2) in last_boxes]
                print(f"areas: min={min(areas):.0f} med={np.median(areas):.0f} max={max(areas):.0f}")
            print(f"FPS: {fps:.1f} | infer: {last_infer_ms:.0f} ms | offset=({BOX_X_OFFSET},{BOX_Y_OFFSET})")
            print(counts_disp, flush=True)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(']'):
            BOX_X_OFFSET += 2
        elif key == ord('['):
            BOX_X_OFFSET -= 2
        elif key == ord('=') or key == ord('+'):
            BOX_Y_OFFSET += 2
        elif key == ord('-'):
            BOX_Y_OFFSET -= 2

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
