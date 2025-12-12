#!/usr/bin/env python3
"""
Simple gaze viewer for AcquisitionsEyeTracker data.

Usage examples:
    python view_gaze.py --subject AcquisitionsEyeTracker/sujet1_f-42e0d11a --video path/to/world_video.mp4
    python view_gaze.py --subject AcquisitionsEyeTracker/sujet1_f-42e0d11a --frames path/to/frames_dir

The script will read `gaze.csv` and `world_timestamps.csv` inside the subject folder
and overlay the gaze point (gaze x [px], gaze y [px]) on each video/frame.
If `world_timestamps.csv` contains one timestamp per frame and its length matches the
video/frame count, it will be used to align gaze samples. Otherwise timestamps are
estimated from video fps or linearly interpolated.
"""
import argparse
import os
import cv2
import numpy as np
import pandas as pd
import math
import logging
import time


def load_world_timestamps(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'timestamp [ns]' in df.columns:
        return df['timestamp [ns]'].values.astype('int64')
    # fallback: use third column
    return df.iloc[:, 2].values.astype('int64')


def load_gaze(path):
    df = pd.read_csv(path)
    # Expected columns: 'timestamp [ns]', 'gaze x [px]', 'gaze y [px]'
    # Try to be permissive with column names
    ts_col = next((c for c in df.columns if 'timestamp' in c.lower()), None)
    x_col = next((c for c in df.columns if 'gaze x' in c.lower() or 'gaze_x' in c.lower() or 'gaze.x' in c.lower()), None)
    y_col = next((c for c in df.columns if 'gaze y' in c.lower() or 'gaze_y' in c.lower() or 'gaze.y' in c.lower()), None)
    if ts_col is None or x_col is None or y_col is None:
        raise ValueError('Could not find expected columns in gaze csv')
    gaze = df[[ts_col, x_col, y_col]].copy()
    gaze.columns = ['ts', 'x', 'y']
    gaze = gaze.dropna(subset=['ts', 'x', 'y'])
    gaze['ts'] = gaze['ts'].astype('int64')
    gaze['x'] = gaze['x'].astype('float32')
    gaze['y'] = gaze['y'].astype('float32')
    return gaze


def find_nearest_gaze(gaze_ts, target_ts):
    # gaze_ts sorted ascending
    idx = np.searchsorted(gaze_ts, target_ts)
    if idx == 0:
        return 0
    if idx >= len(gaze_ts):
        return len(gaze_ts) - 1
    before = idx - 1
    # pick closer of before and idx
    if abs(int(gaze_ts[before]) - int(target_ts)) <= abs(int(gaze_ts[idx]) - int(target_ts)):
        return before
    return idx


def open_frames_dir(frames_dir):
    # list images sorted
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    files = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(exts)]
    files.sort()
    return files


def main():
    p = argparse.ArgumentParser(description='View gaze overlaid on world video or frames')
    p.add_argument('subject', help='Path to subject folder (contains gaze.csv, world_timestamps.csv and the world video or frames)')
    p.add_argument('--video', help='Optional: explicit path to video file. If omitted, script will search inside the subject folder')
    p.add_argument('--affiches', help='Path to directory containing poster images (Affiches). If omitted, script will try to find a sibling `Affiches` folder')
    p.add_argument('--scale', type=float, default=1.0, help='Scale video/frame for display')
    p.add_argument('--detect-every', type=int, default=5, help='Run full detection every N frames (default 5); reuse homography in between')
    p.add_argument('--match-scale', type=float, default=0.5, help='Downscale factor for feature matching (default 0.5 = faster)')
    args = p.parse_args()

    subj = args.subject
    gaze_csv = os.path.join(subj, 'gaze.csv')
    world_ts_csv = os.path.join(subj, 'world_timestamps.csv')
    if not os.path.exists(gaze_csv):
        raise FileNotFoundError(f'Could not find {gaze_csv}')

    gaze = load_gaze(gaze_csv)
    gaze_ts = gaze['ts'].values

    world_ts = load_world_timestamps(world_ts_csv)

    use_video = False
    frames_list = None
    cap = None

    # If user explicitly provided a video path, prefer it. Otherwise try to find a video
    # or frames inside the subject folder.
    video_path = None
    if args.video:
        video_path = args.video
    else:
        # search subject folder for common video extensions
        video_exts = ('.mp4', '.avi', '.mov', '.mkv')
        try:
            found_videos = [os.path.join(subj, f) for f in os.listdir(subj) if f.lower().endswith(video_exts)]
        except Exception:
            found_videos = []
        if len(found_videos) > 0:
            found_videos.sort()
            video_path = found_videos[0]

    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError('Could not open video')
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        use_video = True
    else:
        # look for frames inside the subject folder or a subfolder named 'frames' or 'world_frames'
        candidate_dirs = [subj, os.path.join(subj, 'frames'), os.path.join(subj, 'world_frames'), os.path.join(subj, 'world')]
        frames_list = []
        for d in candidate_dirs:
            if os.path.isdir(d):
                frames_list = open_frames_dir(d)
                if frames_list:
                    break
        if len(frames_list) == 0:
            # also check for image files directly in subject folder
            frames_list = open_frames_dir(subj)

        if len(frames_list) == 0:
            raise RuntimeError('No video or frames found inside subject folder; provide --video or add frames')
        frame_count = len(frames_list)
        fps = 30.0
        tmp = cv2.imread(frames_list[0])
        if tmp is None:
            raise RuntimeError('Could not read first frame image')
        height, width = tmp.shape[:2]

    # load camera intrinsics for undistortion if available
    undistort_maps = None
    scene_cam_path = os.path.join(subj, 'scene_camera.json')
    if os.path.exists(scene_cam_path):
        try:
            import json
            with open(scene_cam_path, 'r') as fh:
                scene = json.load(fh)
            cam = np.array(scene.get('camera_matrix', scene.get('cameraMatrix', [])), dtype=np.float64)
            dist = np.array(scene.get('distortion_coefficients', scene.get('distortionCoefficients', [])), dtype=np.float64)
            # handle nested list (sometimes stored as [[...]] )
            if dist.ndim > 1 and dist.shape[0] == 1:
                dist = dist[0]
            if cam.size == 9 and cam.shape != (3, 3):
                cam = cam.reshape((3, 3))
            if cam.shape == (3, 3) and dist.size >= 4:
                # compute optimal new camera matrix and undistort maps
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam, dist, (width, height), 1, (width, height))
                map1, map2 = cv2.initUndistortRectifyMap(cam, dist, None, newcameramtx, (width, height), cv2.CV_16SC2)
                undistort_maps = (map1, map2)
        except Exception:
            undistort_maps = None

    # Prepare feature detector and matcher: prefer SIFT+FLANN, fallback to ORB+BF
    try:
        detector_sift = cv2.SIFT_create()
        detector = detector_sift
        use_sift = True
        flann_index_kdtree = 1
        index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    except Exception:
        detector = cv2.ORB_create(1000)
        use_sift = False
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Logging setup (DEBUG by default to help diagnose matching issues)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    alg = 'SIFT+FLANN' if use_sift else 'ORB+BF'
    logger.info(f'Gaze viewer starting. Algorithm: {alg}; detect_every={args.detect_every}; match_scale={args.match_scale}')

    # Load poster templates (Affiches)
    def find_affiches_dir(start):
        # search upwards for an 'Affiches' directory or sibling at repo root
        cur = os.path.abspath(start)
        for _ in range(5):
            candidate = os.path.join(cur, 'Affiches')
            if os.path.isdir(candidate):
                return candidate
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        # fallback: sibling of subject parent
        parent = os.path.dirname(start)
        sibling = os.path.join(parent, 'Affiches')
        if os.path.isdir(sibling):
            return sibling
        return None

    def load_affiches(affiches_dir):
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
            kp, des = detector.detectAndCompute(gray, None)
            # create a small thumbnail for fast template checks
            max_thumb = 200
            h, w = img.shape[:2]
            scale_t = min(1.0, max_thumb / max(w, h))
            if scale_t < 1.0:
                thumb = cv2.resize(gray, (int(w*scale_t), int(h*scale_t)))
            else:
                thumb = gray.copy()
            posters.append({'name': f, 'img': img, 'kp': kp, 'des': des, 'shape': img.shape, 'thumb': thumb})
            logger.debug(f'Loaded poster: {f}, keypoints={len(kp)}, descriptors={None if des is None else des.shape}')
        return posters

    affiches_dir = args.affiches if args.affiches else find_affiches_dir(subj)
    posters = load_affiches(affiches_dir)
    if posters:
        logger.info(f'Loaded {len(posters)} poster(s) from: {affiches_dir}')
        logger.debug('Posters: ' + ', '.join([p['name'] for p in posters]))
    else:
        logger.info('No posters loaded (Affiches not found or empty)')

    # cache for detected homographies (avoid re-detecting every frame)
    poster_cache = {'best_poster': None, 'last_frame': -1}

    # decide timestamp per frame
    if world_ts is not None and len(world_ts) >= 2:
        # If lengths match, use them directly; otherwise we'll interpolate
        if len(world_ts) == frame_count:
            frame_ts = world_ts
        else:
            # linear interpolation of timestamps across frames
            start = int(world_ts[0])
            end = int(world_ts[-1])
            frame_idx = np.arange(frame_count)
            frame_ts = (start + (end - start) * (frame_idx / max(1, frame_count - 1))).astype('int64')
    else:
        # estimate from video fps
        start = int(gaze_ts[0]) if len(gaze_ts) > 0 else 0
        frame_ts = (start + (np.arange(frame_count) * (1e9 / fps))).astype('int64')

    gaze_x = gaze['x'].values
    gaze_y = gaze['y'].values

    win = 'Gaze Viewer'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    paused = False
    frame_idx = 0

    while True:
        if not paused:
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

        # compute timestamp for this frame
        if frame_idx >= len(frame_ts):
            break
        ts = frame_ts[frame_idx]

        # find nearest gaze sample
        g_idx = find_nearest_gaze(gaze_ts, ts)
        x = int(round(gaze_x[g_idx]))
        y = int(round(gaze_y[g_idx]))

        # Log frame / gaze info at INFO occasionally and DEBUG otherwise
        if frame_idx % max(1, args.detect_every) == 0:
            logger.info(f'Frame {frame_idx}/{frame_count-1} ts={ts} gaze_idx={g_idx} gaze=({x},{y})')
        else:
            logger.debug(f'Frame {frame_idx} ts={ts} gaze_idx={g_idx}')

        # detect posters in this frame (feature matching + homography) OR track a previously detected poster
        poster_hits = []
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Tracking parameters
        MIN_GOOD_MATCHES = 4   # minimal number of good matches to consider a poster detected
        MIN_POLY_AREA = 1000    # if detected polygon area is larger than this, consider it a solid detection
        MIN_TRACK_AREA = 800   # minimal area to keep tracking

        # Determine whether we should run a fresh detection: prefer detection candidates that enclose the gaze
        tracking = poster_cache.get('tracked')
        best_poster = poster_cache.get('best_poster')
        gaze_inside_tracked = False
        if best_poster is not None:
            try:
                gaze_inside_tracked = cv2.pointPolygonTest(best_poster['poly'], (x, y), False) >= 0
            except Exception:
                gaze_inside_tracked = False

        # If no tracker or gaze is outside the tracked polygon -> run detection now
        # To avoid running expensive detection every frame, require either no tracker
        # or (gaze outside AND frame_idx % detect_every == 0)
        detection_needed = False
        if posters:
            if tracking is None:
                detection_needed = True
            else:
                if not gaze_inside_tracked and (frame_idx % args.detect_every == 0):
                    detection_needed = True

        if detection_needed:
            t0 = time.time()
            scale = args.match_scale
            if scale < 1.0:
                gray_frame_small = cv2.resize(gray_frame, (int(gray_frame.shape[1]*scale), int(gray_frame.shape[0]*scale)))
            else:
                gray_frame_small = gray_frame
            kp_frame, des_frame = detector.detectAndCompute(gray_frame_small, None)
            if kp_frame is None:
                kp_frame = []
            logger.debug(f'Detection run: frame_idx={frame_idx} keypoints_frame={len(kp_frame)} des_frame={None if des_frame is None else des_frame.shape} des_frame_dtype={None if des_frame is None else des_frame.dtype}')
            if des_frame is not None and len(kp_frame) > 0:
                # Detection optimization: restrict expensive operations to an ROI around the gaze
                ROI_SIZE = 400
                H, W = gray_frame.shape[:2]
                rx0 = max(0, int(x - ROI_SIZE//2))
                ry0 = max(0, int(y - ROI_SIZE//2))
                rx1 = min(W, int(x + ROI_SIZE//2))
                ry1 = min(H, int(y + ROI_SIZE//2))
                roi = gray_frame[ry0:ry1, rx0:rx1]

                # First: find quad candidates in ROI (limit to top N by area)
                try:
                    edges = cv2.Canny(roi, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                except Exception:
                    contours = []
                quad_candidates = []
                for cnt in contours:
                    area_cnt = cv2.contourArea(cnt)
                    if area_cnt < 200:  # smaller threshold because ROI is small
                        continue
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        quad = approx.reshape(4, 2).astype(np.float32)
                        quad_candidates.append((area_cnt, quad))
                quad_candidates.sort(key=lambda x: x[0], reverse=True)
                MAX_QUADS = 8
                quad_candidates = quad_candidates[:MAX_QUADS]

                # quick template-matching pre-check to avoid heavy feature matching
                TM_THRESHOLD = 0.6
                for area_cnt, quad in quad_candidates:
                    # map quad from ROI coords to frame coords
                    quad_frame = quad.copy()
                    quad_frame[:, 0] += rx0
                    quad_frame[:, 1] += ry0
                    for poster in posters:
                        thumb = poster.get('thumb')
                        if thumb is None:
                            continue
                        # template matching: try matching poster thumb on roi (fast)
                        try:
                            res = cv2.matchTemplate(roi, thumb, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, _ = cv2.minMaxLoc(res)
                        except Exception:
                            max_val = 0.0
                        logger.debug(f"Quad-local TM poster={poster['name']} max_val={max_val:.3f}")
                        if max_val < TM_THRESHOLD:
                            continue
                        # warp quad_frame to poster size and do local feature matching (expensive)
                        w_p, h_p = poster['shape'][1], poster['shape'][0]
                        dst = np.array([[0, 0], [w_p, 0], [w_p, h_p], [0, h_p]], dtype=np.float32)
                        try:
                            Mq = cv2.getPerspectiveTransform(quad_frame, dst)
                            warped = cv2.warpPerspective(frame, Mq, (w_p, h_p))
                            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                        except Exception:
                            continue
                        kp_w, des_w = detector.detectAndCompute(warped_gray, None)
                        if des_w is None or len(kp_w) == 0:
                            continue
                        try:
                            matches_local = matcher.knnMatch(poster['des'], des_w, k=2)
                        except Exception:
                            try:
                                rawm = matcher.match(poster['des'], des_w)
                                matches_local = [[m] for m in rawm]
                            except Exception:
                                continue
                        good_local = []
                        for mm in matches_local:
                            if len(mm) < 2:
                                continue
                            m1, m2 = mm[0], mm[1]
                            if m1.distance < 0.75 * m2.distance:
                                good_local.append(m1)
                        logger.debug(f"Local quad test poster={poster['name']} good_local={len(good_local)} area_cnt={area_cnt}")
                        if len(good_local) >= max(8, MIN_GOOD_MATCHES):
                            inliers = len(good_local)
                            projected = quad_frame.reshape(-1, 1, 2)
                            poster_hits.append({'name': poster['name'], 'poly': projected, 'inliers': inliers, 'area': area_cnt, 'gaze_inside': cv2.pointPolygonTest(projected, (x, y), False) >= 0, 'dist': 0.0})
                    if poster_hits:
                        break
                if poster_hits:
                    logger.info(f'Found poster via quad candidates at frame {frame_idx}: {poster_hits[0]["name"]}')
                else:
                    # fallback: global feature matching (but limit posters by cheap thumb TM)
                    posters_to_test = []
                    for poster in posters:
                        thumb = poster.get('thumb')
                        if thumb is None:
                            posters_to_test.append(poster)
                            continue
                        try:
                            res = cv2.matchTemplate(gray_frame, thumb, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, _ = cv2.minMaxLoc(res)
                        except Exception:
                            max_val = 0.0
                        if max_val >= 0.45:  # lower threshold for global prefilter
                            posters_to_test.append(poster)
                    # now run the existing global matching only on posters_to_test
                    for poster in posters_to_test:
                        try:
                            matches = matcher.knnMatch(poster['des'], des_frame, k=2)
                        except Exception:
                            try:
                                raw = matcher.match(poster['des'], des_frame)
                                matches = [[m] for m in raw]
                            except Exception:
                                continue
                        good = []
                        for m in matches:
                            if len(m) < 2:
                                continue
                            m1, m2 = m[0], m[1]
                            if m1.distance < 0.75 * m2.distance:
                                good.append(m1)
                        logger.debug(f"Poster '{poster['name']}' good_matches={len(good)}")
                        if len(good) < MIN_GOOD_MATCHES:
                            continue
                        # proceed with homography using quadrant selection as before
                        kp_list = poster['kp']
                        h, w = poster['shape'][:2]
                        cx, cy = (w / 2.0, h / 2.0)
                        quads = {0: [], 1: [], 2: [], 3: []}
                        for m in good:
                            pt = kp_list[m.queryIdx].pt
                            qx = 0 if pt[0] < cx else 1
                            qy = 0 if pt[1] < cy else 1
                            quad_idx = qy * 2 + qx
                            dx = pt[0] - cx
                            dy = pt[1] - cy
                            dist2 = dx*dx + dy*dy
                            quads[quad_idx].append((dist2, m))
                        K_total = min(len(good), 40)
                        K_per_quad = max(1, K_total // 4)
                        sel = []
                        for qi in range(4):
                            if len(quads[qi]) == 0:
                                continue
                            quads[qi].sort(key=lambda x: x[0])
                            take = min(len(quads[qi]), K_per_quad)
                            sel.extend([item[1] for item in quads[qi][:take]])
                        if len(sel) < 4:
                            def dist_to_center(m):
                                pt = kp_list[m.queryIdx].pt
                                dx = pt[0] - cx
                                dy = pt[1] - cy
                                return dx*dx + dy*dy
                            good_sorted = sorted(good, key=dist_to_center)
                            takeK = max(4, min(len(good_sorted), 20))
                            sel = good_sorted[:takeK]
                        src_pts = np.float32([kp_list[m.queryIdx].pt for m in sel]).reshape(-1, 1, 2)
                        dst_pts_scaled = np.float32([kp_frame[m.trainIdx].pt for m in sel]).reshape(-1, 1, 2)
                        if scale < 1.0:
                            dst_pts = dst_pts_scaled / scale
                        else:
                            dst_pts = dst_pts_scaled
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if M is None or mask is None:
                            continue
                        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                        projected = cv2.perspectiveTransform(corners, M)
                        inliers = int(mask.sum())
                        area = abs(cv2.contourArea(projected))
                        pts2 = projected.reshape(-1, 2)
                        minx, miny = pts2.min(axis=0)
                        maxx, maxy = pts2.max(axis=0)
                        bw = maxx - minx
                        bh = maxy - miny
                        logger.debug(f"Poster '{poster['name']}' projected bbox w={bw:.1f} h={bh:.1f} area={area}")
                        if bw < 10 or bh < 10 or not np.isfinite(area):
                            logger.debug(f"Rejecting poster '{poster['name']}' due to small bbox ({bw:.1f}x{bh:.1f}) or invalid area")
                            continue
                        try:
                            dist_to_gaze = abs(cv2.pointPolygonTest(projected, (x, y), True))
                            gaze_inside = cv2.pointPolygonTest(projected, (x, y), False) >= 0
                        except Exception:
                            dist_to_gaze = float('inf')
                            gaze_inside = False
                        logger.info(f"Poster '{poster['name']}' detected, inliers={inliers}, area={area}, gaze_inside={gaze_inside}, dist={dist_to_gaze}")
                        poster_hits.append({'name': poster['name'], 'poly': projected, 'inliers': inliers, 'area': area, 'gaze_inside': gaze_inside, 'dist': dist_to_gaze})
                for poster in posters:
                    if poster.get('des') is None or len(poster.get('des')) == 0:
                        continue
                    try:
                        des_dtype = poster['des'].dtype
                    except Exception:
                        des_dtype = None
                    logger.debug(f"Poster '{poster['name']}' des dtype={des_dtype}")
                    try:
                        matches = matcher.knnMatch(poster['des'], des_frame, k=2)
                    except Exception:
                        try:
                            raw = matcher.match(poster['des'], des_frame)
                            matches = [[m] for m in raw]
                        except Exception:
                            continue
                    # ratio test (Lowe)
                    good = []
                    for m in matches:
                        if len(m) < 2:
                            continue
                        m1, m2 = m[0], m[1]
                        if m1.distance < 0.75 * m2.distance:
                            good.append(m1)
                    logger.debug(f"Poster '{poster['name']}' good_matches={len(good)}")
                    if len(good) < MIN_GOOD_MATCHES:
                        continue

                    # select match subset (cover quadrants when possible)
                    kp_list = poster['kp']
                    h, w = poster['shape'][:2]
                    cx, cy = (w / 2.0, h / 2.0)
                    quads = {0: [], 1: [], 2: [], 3: []}
                    for m in good:
                        pt = kp_list[m.queryIdx].pt
                        qx = 0 if pt[0] < cx else 1
                        qy = 0 if pt[1] < cy else 1
                        quad_idx = qy * 2 + qx
                        dx = pt[0] - cx
                        dy = pt[1] - cy
                        dist2 = dx*dx + dy*dy
                        quads[quad_idx].append((dist2, m))
                    K_total = min(len(good), 40)
                    K_per_quad = max(1, K_total // 4)
                    sel = []
                    for qi in range(4):
                        if len(quads[qi]) == 0:
                            continue
                        quads[qi].sort(key=lambda x: x[0])
                        take = min(len(quads[qi]), K_per_quad)
                        sel.extend([item[1] for item in quads[qi][:take]])
                    if len(sel) < 4:
                        def dist_to_center(m):
                            pt = kp_list[m.queryIdx].pt
                            dx = pt[0] - cx
                            dy = pt[1] - cy
                            return dx*dx + dy*dy
                        good_sorted = sorted(good, key=dist_to_center)
                        takeK = max(4, min(len(good_sorted), 20))
                        sel = good_sorted[:takeK]

                    src_pts = np.float32([kp_list[m.queryIdx].pt for m in sel]).reshape(-1, 1, 2)
                    dst_pts_scaled = np.float32([kp_frame[m.trainIdx].pt for m in sel]).reshape(-1, 1, 2)
                    if scale < 1.0:
                        dst_pts = dst_pts_scaled / scale
                    else:
                        dst_pts = dst_pts_scaled
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is None or mask is None:
                        continue
                    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    projected = cv2.perspectiveTransform(corners, M)
                    inliers = int(mask.sum())
                    area = abs(cv2.contourArea(projected))
                    pts2 = projected.reshape(-1, 2)
                    minx, miny = pts2.min(axis=0)
                    maxx, maxy = pts2.max(axis=0)
                    bw = maxx - minx
                    bh = maxy - miny
                    logger.debug(f"Poster '{poster['name']}' projected bbox w={bw:.1f} h={bh:.1f} area={area}")
                    if bw < 10 or bh < 10 or not np.isfinite(area):
                        logger.debug(f"Rejecting poster '{poster['name']}' due to small bbox ({bw:.1f}x{bh:.1f}) or invalid area")
                        continue
                    # compute distance from gaze to polygon (0 if inside)
                    try:
                        dist_to_gaze = abs(cv2.pointPolygonTest(projected, (x, y), True))
                        gaze_inside = cv2.pointPolygonTest(projected, (x, y), False) >= 0
                    except Exception:
                        dist_to_gaze = float('inf')
                        gaze_inside = False
                    logger.info(f"Poster '{poster['name']}' detected, inliers={inliers}, area={area}, gaze_inside={gaze_inside}, dist={dist_to_gaze}")
                    poster_hits.append({'name': poster['name'], 'poly': projected, 'inliers': inliers, 'area': area, 'gaze_inside': gaze_inside, 'dist': dist_to_gaze})

            # pick best poster: prefer ones that contain gaze
            if poster_hits:
                inside_hits = [p for p in poster_hits if p.get('gaze_inside')]
                if inside_hits:
                    inside_hits.sort(key=lambda p: p['inliers'], reverse=True)
                    best = inside_hits[0]
                else:
                    poster_hits.sort(key=lambda p: (p['dist'], -p['inliers']))
                    best = poster_hits[0]
                poster_cache['best_poster'] = best
                poster_cache['last_frame'] = frame_idx
                corners_pts = best['poly'].reshape(-1, 1, 2).astype(np.float32)
                if best['area'] >= MIN_POLY_AREA or best.get('gaze_inside'):
                    poster_cache['tracked'] = {
                        'name': best['name'],
                        'pts': corners_pts,
                        'prev_gray': gray_frame.copy()
                    }
                    logger.info(f"Initialized tracker for poster '{best['name']}' (area={best['area']}, gaze_inside={best.get('gaze_inside')})")
            t1 = time.time()
            logger.debug(f'Detection time for frame {frame_idx}: {(t1-t0):.3f}s')
        else:
            # we have an active tracker and gaze is inside - track corners with LK
            if tracking is not None:
                prev_gray = tracking['prev_gray']
                pts_prev = tracking['pts']
                try:
                    pts_next, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, pts_prev, None, winSize=(15,15), maxLevel=2)
                except Exception as e:
                    pts_next, st = None, None
                    logger.debug(f'LK tracking failed: {e}')
                tracked_ok = False
                if pts_next is not None and st is not None:
                    st = st.reshape(-1)
                    good_count = int(np.count_nonzero(st))
                    if good_count >= 3:
                        new_poly = pts_next.reshape(-1, 1, 2)
                        area = abs(cv2.contourArea(new_poly))
                        try:
                            gaze_inside = cv2.pointPolygonTest(new_poly, (x, y), False) >= 0
                        except Exception:
                            gaze_inside = False
                        if gaze_inside and (area >= MIN_TRACK_AREA or gaze_inside):
                            tracked_ok = True
                            poster_cache['best_poster'] = {'name': tracking['name'], 'poly': new_poly, 'inliers': None}
                            poster_cache['tracked']['pts'] = pts_next.reshape(-1,1,2)
                            poster_cache['tracked']['prev_gray'] = gray_frame.copy()
                            logger.debug(f"Tracked poster '{tracking['name']}' -> area={area}, gaze_inside={gaze_inside}")
                if not tracked_ok:
                    logger.info(f"Lost tracker for poster '{tracking.get('name')}'. Will re-run detection next frame.")
                    poster_cache['tracked'] = None
                    poster_cache['best_poster'] = None

        # overlay posters (draw polygons) and indicate if gaze falls inside
        vis = frame.copy()
        best_poster = poster_cache['best_poster']
        if best_poster is not None:
            poly = best_poster['poly'].astype(int)
            cv2.polylines(vis, [poly.reshape(-1, 2)], True, (0, 255, 0), 3)
            # label
            cx = int(np.mean(poly[:, 0, 0]))
            cy = int(np.mean(poly[:, 0, 1]))
            cv2.putText(vis, f"Poster: {best_poster['name']}", (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # check gaze inside
            inside = cv2.pointPolygonTest(best_poster['poly'], (x, y), False) >= 0
            logger.info(f"Gaze point ({x},{y}) inside poster '{best_poster['name']}' = {inside}")
            if inside:
                cv2.putText(vis, 'LOOKING', (cx - 50, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        
        # overlay gaze
        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.circle(vis, (x, y), 12, (0, 0, 255), -1)
            cv2.circle(vis, (x, y), 18, (0, 0, 255), 2)
        cv2.putText(vis, f'ts={ts}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(vis, f'gaze_idx={g_idx}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        if args.scale != 1.0:
            vis = cv2.resize(vis, (int(vis.shape[1]*args.scale), int(vis.shape[0]*args.scale)))

        cv2.imshow(win, vis)

        key = cv2.waitKey(int(1000 / fps) if not paused else 0) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord(' '):
            paused = not paused
        if not paused:
            frame_idx += 1

    if use_video and cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
