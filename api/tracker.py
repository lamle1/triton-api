"""
ByteTrack — pure Python / NumPy / SciPy implementation.
No external tracking library required.

Paper: ByteTrack: Multi-Object Tracking by Associating Every Detection Box
Key improvements over SimpleIoU:
  - Kalman filter for motion prediction (handles fast motion / missed frames)
  - Two-stage association:
      Stage 1: high-confidence dets matched to active tracks (IoU)
      Stage 2: low-confidence dets matched to lost tracks
  - Track state machine: Tracked → Lost → Removed
  - Retains global_id / ReID slot for cross-camera identity
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Kalman Filter (constant-velocity, 7-dim state: cx,cy,s,r,vx,vy,vs)
# ---------------------------------------------------------------------------

class KalmanFilter:
    """
    State: [cx, cy, s, r, vx, vy, vs]
    cx,cy = bbox centre; s = area; r = aspect ratio (fixed); v* = velocity
    Measurement: [cx, cy, s, r]
    """

    def __init__(self):
        dt = 1.0
        self.F = np.eye(7)          # state transition
        for i in range(4):
            self.F[i, i + 3] = dt

        self.H = np.zeros((4, 7))   # measurement matrix
        self.H[:4, :4] = np.eye(4)

        # Noise matrices (tuned for CCTV / 10-30 FPS)
        self._std_weight_pos = 1.0 / 20
        self._std_weight_vel = 1.0 / 160

    def init(self, measurement):
        """measurement: [cx, cy, s, r]"""
        mean = np.zeros(7)
        mean[:4] = measurement
        std = [
            2 * self._std_weight_pos * measurement[2],
            2 * self._std_weight_pos * measurement[2],
            1e-2,
            1e-5,
            10 * self._std_weight_vel * measurement[2],
            10 * self._std_weight_vel * measurement[2],
            1e-5,
        ]
        cov = np.diag(np.square(std))
        return mean, cov

    def predict(self, mean, cov):
        std = [
            self._std_weight_pos * mean[2],
            self._std_weight_pos * mean[2],
            1e-2,
            1e-5,
            self._std_weight_vel * mean[2],
            self._std_weight_vel * mean[2],
            1e-5,
        ]
        Q = np.diag(np.square(std))
        mean = self.F @ mean
        cov  = self.F @ cov @ self.F.T + Q
        return mean, cov

    def update(self, mean, cov, measurement):
        std = [
            self._std_weight_pos * mean[2],
            self._std_weight_pos * mean[2],
            1e-1,
            1e-5,
        ]
        R = np.diag(np.square(std))
        S = self.H @ cov @ self.H.T + R
        K = cov @ self.H.T @ np.linalg.inv(S)
        innovation = measurement - self.H @ mean
        mean = mean + K @ innovation
        cov  = (np.eye(7) - K @ self.H) @ cov
        return mean, cov


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xyxy_to_cxcysr(bbox):
    """[x1,y1,x2,y2] → [cx,cy,s,r]"""
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return np.array([x1 + w / 2, y1 + h / 2, w * h, w / h])


def _cxcysr_to_xyxy(mean):
    """[cx,cy,s,r,...] → [x1,y1,x2,y2]"""
    s, r = mean[2], mean[3]
    w = np.sqrt(max(1e-6, s * r))
    h = max(1e-6, s / w)
    cx, cy = mean[0], mean[1]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _iou_matrix(bboxes_a, bboxes_b):
    """Return IoU matrix (len_a × len_b)."""
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)))
    a = np.array(bboxes_a, dtype=np.float32)
    b = np.array(bboxes_b, dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.maximum(0, ix2 - ix1)
    ih = np.maximum(0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / union, 0)

    # Compute DIoU penalty
    cx_a = (a[:, 0] + a[:, 2]) / 2.0
    cy_a = (a[:, 1] + a[:, 3]) / 2.0
    cx_b = (b[:, 0] + b[:, 2]) / 2.0
    cy_b = (b[:, 1] + b[:, 3]) / 2.0
    d2 = (cx_a[:, None] - cx_b[None, :]) ** 2 + (cy_a[:, None] - cy_b[None, :]) ** 2

    cx1 = np.minimum(a[:, None, 0], b[None, :, 0])
    cy1 = np.minimum(a[:, None, 1], b[None, :, 1])
    cx2 = np.maximum(a[:, None, 2], b[None, :, 2])
    cy2 = np.maximum(a[:, None, 3], b[None, :, 3])
    c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + 1e-16

    diou = iou - (d2 / c2)
    return diou


def _linear_assignment(cost_matrix, thresh):
    """Hungarian matching; returns (row_idx, col_idx, unmatched_rows, unmatched_cols)."""
    if cost_matrix.size == 0:
        return [], [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_r, matched_c = [], []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matched_r.append(r)
            matched_c.append(c)
    unmatched_r = [i for i in range(cost_matrix.shape[0]) if i not in matched_r]
    unmatched_c = [j for j in range(cost_matrix.shape[1]) if j not in matched_c]
    return matched_r, matched_c, unmatched_r, unmatched_c


# ---------------------------------------------------------------------------
# Track states
# ---------------------------------------------------------------------------
NEW      = 0
TRACKED  = 1
LOST     = 2
REMOVED  = 3


class STrack:
    _next_id = 1
    _kf = KalmanFilter()

    def __init__(self, bbox_xyxy, score):
        self.local_id    = STrack._next_id
        self.track_id    = STrack._next_id
        STrack._next_id += 1
        self.global_id   = None          # set by Re-ID pipeline
        self.score       = score
        self.state       = NEW
        self.hits        = 1
        self.age         = 1
        self.time_since_update = 0
        self.crops       = []            # stores up to 3 high-quality keyframe crops

        meas = _xyxy_to_cxcysr(bbox_xyxy)
        self.mean, self.cov = STrack._kf.init(meas)
        self._bbox_xyxy = np.array(bbox_xyxy, dtype=np.float32)

    def add_crop_candidate(self, frame_jpeg, bbox, score, frame_timestamp=None):
        """
        Evaluate and store high-quality keyframes for background Re-ID.
        """
        # Collect crops on:
        # - The very first hit (to guarantee we have a fallback crop for fast objects)
        # - Every 2nd hit for the first 10 hits (fast tracking window)
        # - Every 5th hit thereafter (slow tracking window to save CPU)
        is_candidate_frame = (
            self.hits == 1 or
            (self.hits <= 10 and self.hits % 2 == 0) or
            (self.hits > 10 and self.hits % 5 == 0)
        )
        if not is_candidate_frame:
            return

        x1, y1, x2, y2 = map(int, bbox)
        w, h = x2 - x1, y2 - y1
        if w < 15 or h < 15:
            return

        try:
            import cv2
            nparr = np.frombuffer(frame_jpeg, np.uint8)
            full_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if full_img is None or full_img.size == 0:
                return
            
            crop = full_img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size == 0:
                return

            sharpness = max(1.0, float(cv2.Laplacian(crop, cv2.CV_64F).var()))
                
            # Normalize size factor using square root of area
            size_factor = float(np.sqrt(w * h))
            
            # Border Penalty: mild penalty for crops touching the extreme edge (5px margin)
            full_h, full_w = full_img.shape[:2]
            border_penalty = 1.0
            if x1 <= 5 or y1 <= 5 or x2 >= full_w - 5 or y2 >= full_h - 5:
                border_penalty = 0.7

            quality = float(size_factor * score * sharpness * border_penalty)
            
            import time
            candidate = {
                "quality": quality,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(score),
                "frame_jpeg": frame_jpeg,
                "timestamp": frame_timestamp if frame_timestamp is not None else time.time()
            }
            
            self.crops.append(candidate)
            self.crops.sort(key=lambda c: c["quality"], reverse=True)
            self.crops = self.crops[:3]
        except Exception as e:
            print(f"[BYTETracker] add_crop_candidate error: {e}")

    # ---- Kalman interface ----
    def predict(self):
        self.mean, self.cov = STrack._kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox_xyxy, score):
        meas = _xyxy_to_cxcysr(bbox_xyxy)
        self.mean, self.cov = STrack._kf.update(self.mean, self.cov, meas)
        self._bbox_xyxy = np.array(bbox_xyxy, dtype=np.float32)
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        self.state = TRACKED

    def predicted_bbox(self):
        """Predicted bbox after last predict() call."""
        return _cxcysr_to_xyxy(self.mean)

    @property
    def bbox(self):
        return self._bbox_xyxy.tolist()


# ---------------------------------------------------------------------------
# BYTETracker
# ---------------------------------------------------------------------------

class BYTETracker:
    """
    ByteTrack tracker.
    update() accepts list of dicts: {'bbox': [x1,y1,x2,y2], 'score': float}
    OR just list of [x1,y1,x2,y2] (scores default to 1.0 for backward compat).

    Returns list of dicts (same order / length as input dets, None for unmatched):
      {'local_id': int, 'global_id': str|None, 'bbox': [x1,y1,x2,y2],
       'hits': int, 'score': float}
    """

    def __init__(
        self,
        track_thresh=0.45,   # min score for stage-1 high-conf dets
        match_thresh=0.8,    # IoU cost threshold (1 - iou) for assignment
        max_lost=30,         # frames before LOST track is removed
        min_hits=1,          # hits before track considered confirmed
    ):
        self.track_thresh  = track_thresh
        self.match_thresh  = match_thresh
        self.max_lost      = max_lost
        self.min_hits      = min_hits

        self.tracked:  list[STrack] = []
        self.lost:     list[STrack] = []
        self.removed:  list[STrack] = []

    # ------------------------------------------------------------------ #
    def update(self, detections):
        """
        detections: list of [x1,y1,x2,y2]  (scores default 1.0)
                 OR list of {'bbox':[x1,y1,x2,y2], 'score': float}
        Returns list, same length, each element dict or None.
        """
        if not detections:
            # Mark everything lost
            for t in self.tracked:
                t.state = LOST
                t.time_since_update += 1
            self.lost   += self.tracked
            self.tracked = []
            self._prune_lost()
            return []

        # Normalise input
        dets_xyxy, dets_score = [], []
        for d in detections:
            if isinstance(d, dict):
                dets_xyxy.append(d["bbox"])
                dets_score.append(float(d.get("score", 1.0)))
            else:
                dets_xyxy.append(list(d))
                dets_score.append(1.0)

        dets_xyxy  = [np.array(b, dtype=np.float32) for b in dets_xyxy]
        dets_score = np.array(dets_score, dtype=np.float32)

        # Split into high / low confidence
        hi_mask = dets_score >= self.track_thresh
        lo_mask = ~hi_mask
        hi_idx  = np.where(hi_mask)[0].tolist()
        lo_idx  = np.where(lo_mask)[0].tolist()

        # Predict all active tracks
        all_tracks = self.tracked + self.lost
        for t in all_tracks:
            t.predict()

        active = [t for t in self.tracked]

        result = [None] * len(detections)

        # ---- Stage 1: high-conf dets ↔ active tracks ----
        matched_t, matched_d, unmatched_t, unmatched_d_hi = self._match(
            active, [dets_xyxy[i] for i in hi_idx]
        )
        for ti, di in zip(matched_t, matched_d):
            real_di = hi_idx[di]
            active[ti].update(dets_xyxy[real_di], dets_score[real_di])
            result[real_di] = self._make_result(active[ti])

        unmatched_active = [active[i] for i in unmatched_t]
        remaining_hi_det_idx = [hi_idx[i] for i in unmatched_d_hi]

        # ---- Stage 2: low-conf dets ↔ unmatched active tracks ----
        if lo_idx and unmatched_active:
            matched_t2, matched_d2, unmatched_t2, _ = self._match(
                unmatched_active, [dets_xyxy[i] for i in lo_idx]
            )
            for ti, di in zip(matched_t2, matched_d2):
                real_di = lo_idx[di]
                unmatched_active[ti].update(dets_xyxy[real_di], dets_score[real_di])
                result[real_di] = self._make_result(unmatched_active[ti])
            unmatched_active = [unmatched_active[i] for i in unmatched_t2]

        # ---- Unmatched active → LOST (try against lost tracks first) ----
        # Stage 3: remaining hi-conf dets ↔ lost tracks
        if remaining_hi_det_idx and self.lost:
            lost_only = [t for t in self.lost]
            matched_t3, matched_d3, _, unmatched_d3 = self._match(
                lost_only, [dets_xyxy[i] for i in remaining_hi_det_idx]
            )
            for ti, di in zip(matched_t3, matched_d3):
                real_di = remaining_hi_det_idx[di]
                lost_only[ti].update(dets_xyxy[real_di], dets_score[real_di])
                lost_only[ti].state = TRACKED
                self.lost.remove(lost_only[ti])
                self.tracked.append(lost_only[ti])
                result[real_di] = self._make_result(lost_only[ti])
            remaining_hi_det_idx = [remaining_hi_det_idx[i] for i in unmatched_d3]

        # ---- Mark unmatched active as LOST ----
        for t in unmatched_active:
            t.state = LOST
            t.time_since_update = getattr(t, "time_since_update", 0)
            if t in self.tracked:
                self.tracked.remove(t)
            self.lost.append(t)

        # ---- Spawn new tracks for remaining unmatched high-conf dets ----
        for di in remaining_hi_det_idx:
            t = STrack(dets_xyxy[di], dets_score[di])
            t.state = TRACKED
            self.tracked.append(t)
            result[di] = self._make_result(t)

        # ---- Prune dead tracks ----
        self._prune_lost()

        return result

    # ------------------------------------------------------------------ #
    def _match(self, tracks, det_bboxes):
        if not tracks or not det_bboxes:
            return [], [], list(range(len(tracks))), list(range(len(det_bboxes)))
        pred_bboxes = [t.predicted_bbox() for t in tracks]
        iou = _iou_matrix(pred_bboxes, det_bboxes)
        cost = 1.0 - iou
        mr, mc, ur, uc = _linear_assignment(cost, self.match_thresh)
        return mr, mc, ur, uc

    def _make_result(self, t):
        return {
            "local_id":  t.local_id,
            "track_id":  t.track_id,
            "global_id": t.global_id,
            "bbox":      t.bbox,
            "hits":      t.hits,
            "score":     float(t.score),
        }

    def _prune_lost(self):
        keep, remove = [], []
        for t in self.lost:
            if t.time_since_update > self.max_lost:
                t.state = REMOVED
                remove.append(t)
            else:
                keep.append(t)
        self.lost    = keep
        self.removed += remove

    # ------------------------------------------------------------------ #
    # Helper: look up a track by local_id (used by main.py ReID update)
    def get_track(self, local_id) -> "STrack | None":
        for t in self.tracked + self.lost + self.removed:
            if t.local_id == local_id:
                return t
        return None

    get_track_by_local_id = get_track

    def set_global_id(self, local_id, global_id):
        t = self.get_track(local_id)
        if t:
            t.global_id = global_id
