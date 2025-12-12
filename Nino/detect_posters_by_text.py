#!/usr/bin/env python3
"""
Detect posters by OCR text in frames/video and associate the text closest to gaze.

Usage:
    python detect_posters_by_text.py <subject_folder> [--video path] [--affiches path] [--roi-size 400] [--save out.csv]

Requirements:
    pip install opencv-python pytesseract pandas numpy pillow
    Also install Tesseract-OCR on your system and ensure `tesseract` is on PATH.
    On Windows: https://github.com/tesseract-ocr/tesseract

Description:
- Pre-extracts OCR text from poster images in `Affiches` (thumbnailed).
- For each frame (video or frames dir) extracts OCR text boxes (optionally limited to ROI around gaze),
  matches OCR strings to poster texts (string similarity) and reports the best match near gaze.

Output: prints results and optionally saves CSV with columns: frame_idx, ts, gaze_x, gaze_y, detected_poster, ocr_text, score

Note: OCR is relatively slow; tune `--roi-size` and run on a subset if you need speed.
"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd
import pytesseract
import json
from PIL import Image
from difflib import SequenceMatcher


def load_world_timestamps(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'timestamp [ns]' in df.columns:
        return df['timestamp [ns]'].values.astype('int64')
    return df.iloc[:, 2].values.astype('int64')


def find_affiches_dir(start):
    cur = os.path.abspath(start)
    for _ in range(5):
        candidate = os.path.join(cur, 'Affiches')
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    parent = os.path.dirname(start)
    sibling = os.path.join(parent, 'Affiches')
    if os.path.isdir(sibling):
        return sibling
    return None


def ocr_image_to_text(img_gray):
    # img_gray: numpy grayscale image
    pil = Image.fromarray(img_gray)
    text = pytesseract.image_to_string(pil, lang='eng')
    return text.strip()


def ocr_boxes(img_gray, config='--psm 6'):
    # returns list of dicts: left, top, width, height, text, conf, cx, cy
    pil = Image.fromarray(img_gray)
    data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, config=config)
    boxes = []
    n = len(data['level'])
    for i in range(n):
        txt = data['text'][i].strip()
        # 'conf' can be a string, int, float or '-1' depending on tesseract/version
        conf_raw = data['conf'][i]
        conf = -1
        try:
            # try numeric conversion (handles '96', '96.0', 96)
            conf = int(float(conf_raw))
        except Exception:
            conf = -1
        if txt == '' or conf < 30:
            continue
        x = int(data['left'][i])
        y = int(data['top'][i])
        w = int(data['width'][i])
        h = int(data['height'][i])
        cx = x + w/2
        cy = y + h/2
        boxes.append({'text': txt, 'conf': conf, 'left': x, 'top': y, 'w': w, 'h': h, 'cx': cx, 'cy': cy})
    return boxes


def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def load_posters_texts(affiches_dir):
    posters = []
    if not affiches_dir or not os.path.isdir(affiches_dir):
        return posters
    for f in sorted(os.listdir(affiches_dir)):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(affiches_dir, f)
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # optional resize to help OCR on large posters
        h, w = gray.shape[:2]
        scale = 800.0 / max(w, h) if max(w,h) > 800 else 1.0
        if scale < 1.0:
            gray_small = cv2.resize(gray, (int(w*scale), int(h*scale)))
        else:
            gray_small = gray
        text = ocr_image_to_text(gray_small)
        posters.append({'name': f, 'path': path, 'text': text})
    return posters


def find_nearest_gaze(gaze_ts, target_ts):
    idx = np.searchsorted(gaze_ts, target_ts)
    if idx == 0:
        return 0
    if idx >= len(gaze_ts):
        return len(gaze_ts)-1
    before = idx-1
    if abs(int(gaze_ts[before]) - int(target_ts)) <= abs(int(gaze_ts[idx]) - int(target_ts)):
        return before
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument('subject', help='subject folder')
    p.add_argument('--video', help='path to video')
    p.add_argument('--affiches', help='path to Affiches dir')
    p.add_argument('--tesseract-cmd', help='Full path to tesseract executable to override PATH')
    p.add_argument('--visualize', action='store_true', help='Show real-time visualization of OCR and detection')
    p.add_argument('--roi-size', type=int, default=400, help='ROI size around gaze for OCR')
    p.add_argument('--ocr-every', type=int, default=5, help='Run OCR every N frames (default 5) to keep visualization real-time')
    p.add_argument('--save', help='save csv output')
    args = p.parse_args()

    subj = args.subject
    affiches_dir = args.affiches if args.affiches else find_affiches_dir(subj)
    posters = load_posters_texts(affiches_dir)
    print(f'Loaded {len(posters)} posters with OCR text from {affiches_dir}')
    for pinfo in posters:
        print('-', pinfo['name'], '->', repr(pinfo['text'][:80]))

    gaze_csv = os.path.join(subj, 'gaze.csv')
    world_ts_csv = os.path.join(subj, 'world_timestamps.csv')
    if not os.path.exists(gaze_csv):
        raise FileNotFoundError(gaze_csv)
    gaze_df = pd.read_csv(gaze_csv)
    ts_col = next((c for c in gaze_df.columns if 'timestamp' in c.lower()), None)
    x_col = next((c for c in gaze_df.columns if 'gaze x' in c.lower() or 'gaze_x' in c.lower()), None)
    y_col = next((c for c in gaze_df.columns if 'gaze y' in c.lower() or 'gaze_y' in c.lower()), None)
    gaze = gaze_df[[ts_col, x_col, y_col]].dropna()
    gaze.columns = ['ts','x','y']
    gaze_ts = gaze['ts'].values
    gaze_x = gaze['x'].values
    gaze_y = gaze['y'].values

    # allow overriding tesseract executable path from CLI
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    world_ts = load_world_timestamps(world_ts_csv)

    # open video or frames
    video_path = args.video
    if video_path is None:
        # try to find video inside subject
        exts = ('.mp4','.avi','.mov','.mkv')
        found_videos = [os.path.join(subj,f) for f in os.listdir(subj) if f.lower().endswith(exts)]
        video_path = found_videos[0] if found_videos else None

    frames_list = None
    cap = None
    use_video = False
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError('could not open video')
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        use_video = True
    else:
        # frames dir
        candidate_dirs = [subj, os.path.join(subj,'frames'), os.path.join(subj,'world_frames')]
        frames_list = []
        for d in candidate_dirs:
            if os.path.isdir(d):
                exts = ('.png','.jpg','.jpeg')
                frames_list = [os.path.join(d,f) for f in os.listdir(d) if f.lower().endswith(exts)]
                frames_list.sort()
                if frames_list:
                    break
        if len(frames_list)==0:
            raise RuntimeError('no video or frames')
        frame_count = len(frames_list)
        fps = 30.0
        tmp = cv2.imread(frames_list[0])
        height, width = tmp.shape[:2]

    # load camera intrinsics for undistortion if available
    undistort_maps = None
    scene_cam_path = os.path.join(subj, 'scene_camera.json')
    if os.path.exists(scene_cam_path):
        try:
            with open(scene_cam_path, 'r') as fh:
                scene = json.load(fh)
            cam = np.array(scene.get('camera_matrix', scene.get('cameraMatrix', [])), dtype=np.float64)
            dist = np.array(scene.get('distortion_coefficients', scene.get('distortionCoefficients', [])), dtype=np.float64)
            # handle nested list
            if dist.ndim > 1 and dist.shape[0] == 1:
                dist = dist[0]
            if cam.size == 9 and cam.shape != (3, 3):
                cam = cam.reshape((3, 3))
            if cam.shape == (3, 3) and dist.size >= 4:
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam, dist, (width, height), 1, (width, height))
                map1, map2 = cv2.initUndistortRectifyMap(cam, dist, None, newcameramtx, (width, height), cv2.CV_16SC2)
                undistort_maps = (map1, map2)
        except Exception:
            undistort_maps = None

    results = []
    frame_idx = 0
    paused = False
    last_detection = {'poster': None, 'text': None, 'score': 0.0, 'boxes': [], 'best_box': None}
    while True:
        if use_video:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            if frame_idx >= len(frames_list):
                break
            frame = cv2.imread(frames_list[frame_idx])
        # undistort frame if maps available
        if undistort_maps is not None:
            try:
                frame = cv2.remap(frame, undistort_maps[0], undistort_maps[1], interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass
        # compute ts for frame
        if world_ts is not None and len(world_ts)==frame_count:
            ts = int(world_ts[frame_idx])
        else:
            ts = int(gaze_ts[0] + frame_idx * (1e9/fps))
        # find nearest gaze
        g_idx = find_nearest_gaze(gaze_ts, ts)
        gx = int(round(gaze_x[g_idx]))
        gy = int(round(gaze_y[g_idx]))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H,W = gray.shape[:2]
        rs = args.roi_size
        x0 = max(0, gx - rs//2)
        y0 = max(0, gy - rs//2)
        x1 = min(W, gx + rs//2)
        y1 = min(H, gy + rs//2)
        roi = gray[y0:y1, x0:x1]

        do_ocr = (frame_idx % max(1, args.ocr_every) == 0) or (last_detection['poster'] is None)
        detected_poster = last_detection.get('poster')
        detected_text = last_detection.get('text')
        detected_score = last_detection.get('score', 0.0)
        boxes = last_detection.get('boxes', [])
        best_box = last_detection.get('best_box')

        if do_ocr:
            boxes = ocr_boxes(roi, config='--psm 6')
            # translate box centers to full-frame coords
            for b in boxes:
                b['cx_full'] = b['cx'] + x0
                b['cy_full'] = b['cy'] + y0
            # choose best box near gaze (distance)
            best_box = None
            best_box_dist = float('inf')
            for b in boxes:
                dx = (b['cx_full'] - gx)
                dy = (b['cy_full'] - gy)
                d = np.hypot(dx,dy)
                if d < best_box_dist:
                    best_box_dist = d
                    best_box = b
            detected_poster = None
            detected_text = None
            detected_score = 0.0
            if best_box is not None:
                # match box text to posters
                txt = best_box['text']
                best_score = 0.0
                best_p = None
                for pinfo in posters:
                    score = similarity(txt, pinfo['text'])
                    if score > best_score:
                        best_score = score
                        best_p = pinfo
                if best_p is not None and best_score > 0.2:
                    detected_poster = best_p['name']
                    detected_text = txt
                    detected_score = best_score
            # fallback: if no box near gaze, try global OCR limited number of boxes
            if detected_poster is None:
                boxes_full = ocr_boxes(gray, config='--psm 3')
                best_score = 0.0
                best_p = None
                best_box_full = None
                for b in boxes_full:
                    for pinfo in posters:
                        score = similarity(b['text'], pinfo['text'])
                        if score > best_score:
                            best_score = score
                            best_p = pinfo
                            best_box_full = b
                if best_p is not None and best_score > 0.3:
                    detected_poster = best_p['name']
                    detected_text = best_box_full['text']
                    detected_score = best_score
            last_detection = {'poster': detected_poster, 'text': detected_text, 'score': detected_score, 'boxes': boxes, 'best_box': best_box}

        print(f'frame={frame_idx} ts={ts} gaze=({gx},{gy}) poster={detected_poster} text={detected_text!r} score={detected_score:.2f}')
        results.append({'frame': frame_idx, 'ts': ts, 'gaze_x': gx, 'gaze_y': gy, 'poster': detected_poster, 'text': detected_text, 'score': detected_score})

        # Visualization: overlay ROI, OCR boxes, gaze and detected poster
        if args.visualize:
            vis = frame.copy()
            # draw ROI rect
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 200, 0), 2)
            # draw OCR boxes (ROI coords -> frame coords)
            for b in boxes:
                lx = int(b['left'] + x0)
                ly = int(b['top'] + y0)
                rx = int(lx + b['w'])
                by = int(ly + b['h'])
                cv2.rectangle(vis, (lx, ly), (rx, by), (0, 255, 255), 1)
                cv2.putText(vis, f"{b['text']} ({b['conf']})", (lx, ly-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            # highlight best_box
            if best_box is not None:
                bx0 = int(best_box['cx_full'] - best_box['w']/2)
                by0 = int(best_box['cy_full'] - best_box['h']/2)
                bx1 = int(bx0 + best_box['w'])
                by1 = int(by0 + best_box['h'])
                cv2.rectangle(vis, (bx0, by0), (bx1, by1), (0, 0, 255), 2)
                cv2.putText(vis, f"MATCH: {detected_poster} ({detected_score:.2f})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            # draw gaze
            if 0 <= gx < W and 0 <= gy < H:
                cv2.circle(vis, (gx, gy), 10, (0,0,255), -1)
                cv2.circle(vis, (gx, gy), 16, (0,0,255), 2)
            cv2.putText(vis, f'frame={frame_idx} ts={ts}', (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.imshow('detect_posters_by_text', vis)
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('q') or key == 27:
                break
            if key == ord(' '):
                paused = not paused

        frame_idx += 1

    if use_video and cap is not None:
        cap.release()
    if args.save:
        df = pd.DataFrame(results)
        df.to_csv(args.save, index=False)
        print('Saved results to', args.save)


if __name__ == '__main__':
    main()
