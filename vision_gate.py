import time
import base64
from typing import Optional, Tuple
import cv2
import numpy as np
import mss
from config import CONFIG


class JevStateClassifier:
    """
    Wrapper for the local/lightweight Jev classifier tooling.
    Evaluates whether a meaningful semantic game-state transition occurred
    between the reference frame and the candidate frame.
    """
    def __init__(self, threshold: float = CONFIG.JEV_CONFIDENCE_THRESHOLD):
        self.threshold = threshold
        self.reference_hist: Optional[np.ndarray] = None

    def predict_transition_confidence(self, frame_bgr: np.ndarray, game_mode: str) -> float:
        """
        Calculates probability [0.0 - 1.0] that the board/table state changed.
        Replace the internal heuristic scoring block with your local Jev model binding
        (e.g., `jev.classify(frame_bgr, prompt="Has the game state changed?")`).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Focus ROI on the center 70% of the frame where cards/board live
        h, w = gray.shape
        roi = gray[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
        
        # Compute normalized color/intensity histogram + edge density
        hist = cv2.calcHist([roi], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)

        if self.reference_hist is None:
            self.reference_hist = hist
            return 1.0  # Always trigger on the first valid settled frame

        # Bhattacharyya distance measures distribution shift (0.0 = identical, 1.0 = distinct)
        confidence = float(cv2.compareHist(self.reference_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        
        if confidence >= self.threshold:
            self.reference_hist = hist
            
        return round(confidence, 4)


class VisionGate:
    """
    Captures frames from the macOS WhatsApp desktop window and gates downstream
    vision API calls using kinematic settling + Jev classification.
    """
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = {
            "top": CONFIG.WINDOW_TOP,
            "left": CONFIG.WINDOW_LEFT,
            "width": CONFIG.WINDOW_WIDTH,
            "height": CONFIG.WINDOW_HEIGHT,
        }
        self.jev = JevStateClassifier()
        self.prev_gray: Optional[np.ndarray] = None
        self.in_motion: bool = False

    def grab_frame(self) -> np.ndarray:
        sct_img = self.sct.grab(self.monitor)
        frame = np.array(sct_img)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def poll_for_state_change(self) -> Tuple[bool, Optional[str], float]:
        """
        Polls the capture region.
        Returns: (triggered: bool, base64_jpeg: Optional[str], jev_confidence: float)
        """
        frame = self.grab_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray_blur
            return False, None, 0.0

        # Calculate pixel-level motion delta to detect hands entering/leaving frame
        frame_delta = cv2.absdiff(self.prev_gray, gray_blur)
        motion_ratio = float(np.count_nonzero(frame_delta > 25)) / frame_delta.size
        self.prev_gray = gray_blur

        # State 1: High motion detected (hand dealing card or moving chess piece)
        if motion_ratio > CONFIG.MOTION_SETTLE_CEILING:
            self.in_motion = True
            return False, None, 0.0

        # State 2: Motion just settled, or subtle static change occurred
        if self.in_motion or (CONFIG.MOTION_SETTLE_FLOOR <= motion_ratio <= CONFIG.MOTION_SETTLE_CEILING):
            self.in_motion = False
            confidence = self.jev.predict_transition_confidence(frame, CONFIG.GAME_MODE)
            
            if confidence >= CONFIG.JEV_CONFIDENCE_THRESHOLD:
                _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                b64_img = base64.b64encode(buffer).decode("utf-8")
                return True, b64_img, confidence
                
            return False, None, confidence

        return False, None, 0.0
