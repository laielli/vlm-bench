from typing import List, Tuple, Dict, Optional
import numpy as np

def select_frames(video_path: str, max_frames: int = 32, strategy: str = "uniform",
                  fps: Optional[float] = None, stride: Optional[int] = None) -> Tuple[List[np.ndarray], Dict]:
    """Return a list of HxWxC RGB frames (np.uint8) and metadata.
    Prefers decord; falls back to OpenCV. Deterministic under fixed seed (selection is pure).
    """
    backend = None
    vr = None
    total = 0
    frames: List[np.ndarray] = []
    indices: List[int] = []

    # Try decord
    try:
        import decord
        decord.bridge.set_bridge('native')
        vr = decord.VideoReader(video_path)
        backend = "decord"
        total = len(vr)
        if strategy == "uniform":
            if total <= max_frames:
                indices = list(range(total))
            else:
                step = total / max_frames
                indices = [int(i * step) for i in range(max_frames)]
        elif strategy == "stride" and stride:
            indices = list(range(0, total, stride))[:max_frames]
        else:
            if total <= max_frames:
                indices = list(range(total))
            else:
                step = total / max_frames
                indices = [int(i * step) for i in range(max_frames)]
        for idx in indices:
            img = vr[idx].asnumpy()
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            frames.append(img)
        meta = {
            "backend": backend,
            "num_frames_total": int(total),
            "selected_indices": [int(x) for x in indices],
            "strategy": strategy,
            "fps_used": fps
        }
        return frames, meta
    except Exception:
        pass

    # Fallback: OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        backend = "opencv"
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if strategy == "uniform" and total > 0:
            if total <= max_frames:
                indices = list(range(total))
            else:
                step = total / max_frames
                indices = [int(i * step) for i in range(max_frames)]
        elif strategy == "stride" and stride:
            indices = list(range(0, total, stride))[:max_frames]
        else:
            indices = list(range(min(total, max_frames)))
        frames_map = set(indices)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if i in frames_map:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            i += 1
            if len(frames) >= max_frames:
                break
        cap.release()
        meta = {
            "backend": backend,
            "num_frames_total": int(total),
            "selected_indices": [int(x) for x in indices[:len(frames)]],
            "strategy": strategy,
            "fps_used": fps
        }
        return frames, meta
    except Exception as e:
        raise RuntimeError(f"Video decoding failed for {video_path}: {e}")
