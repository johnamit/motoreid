"""
MotoGP Team Detection & Tracking Pipeline
==========================================

Deep Re-Identification & State Estimation for High-Velocity Agents

This system demonstrates Foundation Model-based Re-ID using DINOv3 embeddings
to maintain identity permanence across occlusions, camera cuts, and long temporal gaps.

Architecture:
    1. Detection: YOLOv8 (Spatial localization)
    2. Embedding: DINOv3 ViT-S/16 (Semantic representation)
    3. Association: ByteTrack (Short-term) + Cosine Similarity Memory Bank (Long-term)
    4. State Estimation: EMA embedding smoothing + Velocity-based position prediction

Key Features:
    - Label locking after high-confidence agreement
    - Re-ID matching via combined visual + spatial similarity
    - Position trajectory tracking with velocity prediction
    - Global constraint enforcement (max 2 bikes per team)
    - Visual debug overlay (trajectories, Re-ID events)
"""

import os
import sys
import cv2
import torch
import joblib
import logging
import argparse
import numpy as np
from collections import deque, Counter
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from scipy.spatial.distance import cosine
from tqdm import tqdm
from ultralytics import YOLO
from torchvision.transforms import v2

# Add dinov3 repo to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dinov3'))
from dinov3.hub.backbones import dinov3_vits16


# =============================================================================
# Configuration
# =============================================================================

TEAM_COLORS = {
    'aprilia_factory':    (0, 255, 0),      # Green
    'aprilia_trackhouse': (0, 200, 100),    # Teal
    'ducati_lenovo':      (0, 0, 255),      # Red
    'ducati_gresini':     (0, 100, 255),    # Orange-Red
    'ducati_vr46':        (0, 255, 255),    # Yellow
    'honda_hrc':          (0, 0, 0),        # Black
    'honda_lcr':          (200, 200, 200),  # Light Gray
    'ktm_factory':        (0, 165, 255),    # Orange
    'ktm_tech3':          (0, 140, 255),    # Dark Orange
    'yamaha_monster':     (255, 0, 0),      # Blue
    'yamaha_pramac':      (255, 100, 100),  # Light Blue
    'unknown':            (128, 128, 128),  # Gray
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="MotoGP Team Detection & Re-ID Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    parser.add_argument("--source", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--output", type=str, default="data/output/annotated_races/final_race.mp4",
                        help="Path to save annotated video")
    parser.add_argument("--log", type=str, default="data/output/annotated_races/logs/pipeline.log",
                        help="Path to save log file")

    # Model Paths
    parser.add_argument("--yolo_weights", type=str, default="yolov8m.pt",
                        help="YOLOv8 weights")
    parser.add_argument("--classifier_path", type=str,
                        default="runs/classifier/dinov3_identity_model.pkl",
                        help="Path to trained team classifier")
    parser.add_argument("--dino_weights", type=str,
                        default="models/DINO/dinov3_vits16_pretrain_lvd1689m.pth",
                        help="Path to DINOv3 backbone weights")

    # Detection
    parser.add_argument("--conf_thresh", type=float, default=0.25,
                        help="YOLO confidence threshold")
    parser.add_argument("--target_class", type=int, default=3,
                        help="COCO class ID (3 = motorcycle)")
    parser.add_argument("--min_box_size", type=int, default=20,
                        help="Minimum bounding box dimension")

    # Feature Extraction
    parser.add_argument("--img_size", type=int, default=224,
                        help="DINOv3 input size")
    parser.add_argument("--embedding_ema", type=float, default=0.9,
                        help="EMA weight for embedding smoothing (higher = more history)")

    # Tracking & Re-ID
    parser.add_argument("--buffer_size", type=int, default=30,
                        help="Frames for voting history")
    parser.add_argument("--lock_threshold", type=float, default=0.85,
                        help="Agreement ratio to lock label")
    parser.add_argument("--reid_visual_thresh", type=float, default=0.80,
                        help="Cosine similarity threshold for Re-ID")
    parser.add_argument("--reid_spatial_thresh", type=float, default=300,
                        help="Max pixel distance for spatial Re-ID constraint")
    parser.add_argument("--lost_timeout", type=int, default=5,
                        help="Frames before marking track as lost")
    parser.add_argument("--forgotten_timeout", type=int, default=300,
                        help="Frames before forgetting lost track")
    parser.add_argument("--trajectory_length", type=int, default=50,
                        help="Number of positions to keep for trajectory")

    # Visualization
    parser.add_argument("--draw_trajectory", action="store_true",
                        help="Draw position history trail")
    parser.add_argument("--draw_reid_events", action="store_true",
                        help="Flash Re-ID match events on screen")
    parser.add_argument("--reid_flash_frames", type=int, default=30,
                        help="Frames to display Re-ID event")

    # Performance
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth frame")

    return parser.parse_args()


def setup_logger(log_path):
    """Configure logging to file and console."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def make_transform(img_size=224):
    """Create DINOv3 preprocessing transform."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


# =============================================================================
# Track State Management (The "Memory" of each bike)
# =============================================================================

@dataclass
class ReIDEvent:
    """Records a Re-ID match for visualization."""
    old_id: int
    new_id: int
    similarity: float
    frames_remaining: int


class TrackState:
    """
    Maintains the complete state of a tracked bike.
    
    This class implements:
        - EMA-smoothed embedding (visual fingerprint)
        - Position trajectory with velocity prediction
        - Label voting with confidence locking
    """

    def __init__(self, track_id: int, initial_embedding: np.ndarray, 
                 initial_position: Tuple[int, int], buffer_size: int = 30,
                 lock_threshold: float = 0.85, trajectory_length: int = 50,
                 embedding_ema: float = 0.9):
        self.id = track_id
        self.original_id = track_id  # For Re-ID tracking
        
        # Visual identity (The "Fingerprint")
        self.embedding = initial_embedding.copy()
        self.embedding_ema = embedding_ema
        
        # Position state (For spatial Re-ID + trajectory viz)
        self.positions = deque([initial_position], maxlen=trajectory_length)
        
        # Label state
        self.label_history = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.lock_threshold = lock_threshold
        self.locked_label: Optional[str] = None
        
        # Lifecycle
        self.frames_since_seen = 0
        self.total_frames_tracked = 0
        self.is_active = True

    def update(self, label: str, embedding: np.ndarray, 
               position: Tuple[int, int]) -> str:
        """
        Update track with new observation.
        
        Args:
            label: Predicted team label
            embedding: Raw DINO feature vector
            position: (x, y) center of bounding box
            
        Returns:
            Final label (locked or voted)
        """
        self.frames_since_seen = 0
        self.total_frames_tracked += 1
        self.is_active = True
        
        # Update embedding with EMA (smooths out noise/glare)
        # embedding_new = alpha * embedding_old + (1-alpha) * embedding_current
        self.embedding = (self.embedding_ema * self.embedding + 
                         (1 - self.embedding_ema) * embedding)
        
        # Update position trajectory
        self.positions.append(position)
        
        # If locked, maintain stable label
        if self.locked_label:
            return self.locked_label
        
        # Update label history
        self.label_history.append(label)
        
        # Check for lock condition
        if len(self.label_history) >= self.buffer_size:
            most_common, count = Counter(self.label_history).most_common(1)[0]
            agreement = count / len(self.label_history)
            
            if agreement >= self.lock_threshold:
                self.locked_label = most_common
                logging.info(f"🔒 Track {self.id} locked to {self.locked_label} "
                           f"({agreement:.0%} agreement over {self.buffer_size} frames)")
                return self.locked_label
        
        # Return majority vote
        if self.label_history:
            return Counter(self.label_history).most_common(1)[0][0]
        return label

    def predict_next_position(self) -> Tuple[int, int]:
        """
        Predict next position using simple velocity estimation.
        Used for spatial Re-ID constraints.
        """
        if len(self.positions) < 2:
            return self.positions[-1]
        
        # Simple linear velocity (could upgrade to Kalman filter)
        dx = self.positions[-1][0] - self.positions[-2][0]
        dy = self.positions[-1][1] - self.positions[-2][1]
        
        return (self.positions[-1][0] + dx, self.positions[-1][1] + dy)

    def get_current_label(self) -> Optional[str]:
        """Get current best label without updating."""
        if self.locked_label:
            return self.locked_label
        if self.label_history:
            return Counter(self.label_history).most_common(1)[0][0]
        return None

    @property
    def last_position(self) -> Tuple[int, int]:
        return self.positions[-1] if self.positions else (0, 0)


class ReIDManager:
    """
    Global Re-Identification & State Management System.
    
    Implements:
        - Active track management
        - Memory bank for lost tracks
        - Combined visual + spatial Re-ID matching
        - Team constraint enforcement (max 2 per team)
        - Metrics tracking for evaluation
    """

    def __init__(self, buffer_size: int = 30, lock_threshold: float = 0.85,
                 reid_visual_thresh: float = 0.80, reid_spatial_thresh: float = 300,
                 lost_timeout: int = 5, forgotten_timeout: int = 300,
                 trajectory_length: int = 50, embedding_ema: float = 0.9):
        
        self.active_tracks: Dict[int, TrackState] = {}
        self.memory_bank: List[TrackState] = []
        self.reid_events: List[ReIDEvent] = []
        
        # Config
        self.buffer_size = buffer_size
        self.lock_threshold = lock_threshold
        self.reid_visual_thresh = reid_visual_thresh
        self.reid_spatial_thresh = reid_spatial_thresh
        self.lost_timeout = lost_timeout
        self.forgotten_timeout = forgotten_timeout
        self.trajectory_length = trajectory_length
        self.embedding_ema = embedding_ema
        
        # Metrics
        self.total_reid_matches = 0
        self.total_new_tracks = 0
        self.total_id_switches = 0  # Would be higher without Re-ID

    def get_or_create_track(self, track_id: int, embedding: np.ndarray,
                            position: Tuple[int, int]) -> Tuple[TrackState, Optional[ReIDEvent]]:
        """
        Get existing track or create new one with Re-ID matching.
        
        Args:
            track_id: ByteTrack assigned ID
            embedding: DINO feature vector
            position: (x, y) center position
            
        Returns:
            Tuple of (TrackState, ReIDEvent or None)
        """
        reid_event = None
        
        # Case 1: Known active track
        if track_id in self.active_tracks:
            return self.active_tracks[track_id], None
        
        # Case 2: Check memory bank for Re-ID match
        best_match = None
        best_visual_score = self.reid_visual_thresh
        best_combined_score = 0
        
        for ghost in self.memory_bank:
            # Skip if no valid embedding
            if np.allclose(ghost.embedding, 0):
                continue
            
            # Visual similarity (cosine)
            visual_sim = 1 - cosine(embedding, ghost.embedding)
            
            if visual_sim < self.reid_visual_thresh:
                continue
            
            # Spatial constraint (predicted position vs actual)
            pred_pos = ghost.predict_next_position()
            spatial_dist = np.sqrt(
                (position[0] - pred_pos[0])**2 + 
                (position[1] - pred_pos[1])**2
            )
            
            # Combined score (visual similarity weighted by spatial plausibility)
            if spatial_dist < self.reid_spatial_thresh:
                # Normalize spatial distance to 0-1 (closer = higher score)
                spatial_score = 1 - (spatial_dist / self.reid_spatial_thresh)
                combined_score = 0.7 * visual_sim + 0.3 * spatial_score
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_visual_score = visual_sim
                    best_match = ghost
        
        if best_match:
            # Resurrection! Link new ID to old track's state
            self.total_reid_matches += 1
            
            logging.info(f"🔄 Re-ID Match: Track {track_id} ← Ghost {best_match.id} "
                        f"(visual: {best_visual_score:.2f}, combined: {best_combined_score:.2f})")
            
            # Create new state inheriting history from matched ghost
            new_track = TrackState(
                track_id=track_id,
                initial_embedding=best_match.embedding,
                initial_position=position,
                buffer_size=self.buffer_size,
                lock_threshold=self.lock_threshold,
                trajectory_length=self.trajectory_length,
                embedding_ema=self.embedding_ema
            )
            new_track.label_history = best_match.label_history
            new_track.locked_label = best_match.locked_label
            new_track.original_id = best_match.original_id
            new_track.total_frames_tracked = best_match.total_frames_tracked
            
            # Record Re-ID event for visualization
            reid_event = ReIDEvent(
                old_id=best_match.original_id,
                new_id=track_id,
                similarity=best_visual_score,
                frames_remaining=30  # Will be decremented each frame
            )
            self.reid_events.append(reid_event)
            
            self.active_tracks[track_id] = new_track
            self.memory_bank.remove(best_match)
            
            return new_track, reid_event
        
        # Case 3: Brand new track
        self.total_new_tracks += 1
        self.total_id_switches += 1  # This would be an ID switch without Re-ID
        
        new_track = TrackState(
            track_id=track_id,
            initial_embedding=embedding,
            initial_position=position,
            buffer_size=self.buffer_size,
            lock_threshold=self.lock_threshold,
            trajectory_length=self.trajectory_length,
            embedding_ema=self.embedding_ema
        )
        self.active_tracks[track_id] = new_track
        
        return new_track, None

    def enforce_team_constraint(self, detections: List[dict], 
                                max_per_team: int = 2) -> List[dict]:
        """Ensure no team has more than max_per_team bikes."""
        team_counts = Counter(
            d['label'] for d in detections if d['label'] != 'unknown'
        )
        
        for team, count in team_counts.items():
            if count > max_per_team:
                team_dets = [d for d in detections if d['label'] == team]
                # Prioritize locked tracks, then by confidence
                team_dets.sort(key=lambda x: (x['locked'], x['conf']), reverse=True)
                
                for loser in team_dets[max_per_team:]:
                    logging.debug(f"⚠️ Team constraint: Track {loser['track_id']} "
                                 f"({team}) → unknown")
                    loser['label'] = 'unknown'
        
        return detections

    def cleanup(self, seen_track_ids: set):
        """Move stale tracks to memory bank, forget old ghosts."""
        # Mark unseen tracks and move to memory bank if timeout
        for track_id in list(self.active_tracks.keys()):
            track = self.active_tracks[track_id]
            
            if track_id not in seen_track_ids:
                track.frames_since_seen += 1
                track.is_active = False
                
                if track.frames_since_seen > self.lost_timeout:
                    logging.debug(f"📤 Track {track_id} → Memory Bank")
                    self.memory_bank.append(track)
                    del self.active_tracks[track_id]
        
        # Age memory bank and forget old ghosts
        for ghost in self.memory_bank:
            ghost.frames_since_seen += 1
        
        self.memory_bank = [
            g for g in self.memory_bank 
            if g.frames_since_seen < self.forgotten_timeout
        ]
        
        # Age Re-ID events
        for event in self.reid_events:
            event.frames_remaining -= 1
        self.reid_events = [e for e in self.reid_events if e.frames_remaining > 0]

    def get_metrics(self) -> dict:
        """Return tracking metrics for evaluation."""
        return {
            'total_reid_matches': self.total_reid_matches,
            'total_new_tracks': self.total_new_tracks,
            'potential_id_switches_prevented': self.total_reid_matches,
            'active_tracks': len(self.active_tracks),
            'memory_bank_size': len(self.memory_bank),
        }


# =============================================================================
# Pipeline Class
# =============================================================================

class MotoGPPipeline:
    """
    End-to-end pipeline for detecting, tracking, and identifying MotoGP bikes
    with Foundation Model-based Re-ID.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logging.info(f"🚀 Initializing Pipeline on {self.device}")

        self._load_yolo()
        self._load_dino()
        self._load_classifier()

        self.transform = make_transform(args.img_size)
        
        # Initialize Re-ID manager
        self.manager = ReIDManager(
            buffer_size=args.buffer_size,
            lock_threshold=args.lock_threshold,
            reid_visual_thresh=args.reid_visual_thresh,
            reid_spatial_thresh=args.reid_spatial_thresh,
            lost_timeout=args.lost_timeout,
            forgotten_timeout=args.forgotten_timeout,
            trajectory_length=args.trajectory_length,
            embedding_ema=args.embedding_ema
        )

    def _load_yolo(self):
        """Load YOLOv8 detection model."""
        logging.info(f"👁️  Loading YOLOv8: {self.args.yolo_weights}")
        self.yolo = YOLO(self.args.yolo_weights)

    def _load_dino(self):
        """Load DINOv3 feature extraction backbone."""
        logging.info(f"🦖 Loading DINOv3: {self.args.dino_weights}")

        self.dino = dinov3_vits16(pretrained=False)
        state_dict = torch.load(
            self.args.dino_weights,
            map_location=self.device,
            weights_only=True
        )
        self.dino.load_state_dict(state_dict)
        self.dino.to(self.device).eval()

    def _load_classifier(self):
        """Load trained team classifier."""
        logging.info(f"🧠 Loading Classifier: {self.args.classifier_path}")

        data = joblib.load(self.args.classifier_path)
        self.classifier = data['model']
        self.classes = data['classes']

        logging.info(f"   Classes: {self.classes}")

    def extract_features(self, crop: np.ndarray) -> np.ndarray:
        """Extract DINOv3 CLS token features from image crop."""
        img_tensor = self.transform(crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.dino(img_tensor)

            if isinstance(output, dict):
                features = output['x_norm_clstoken']
            elif output.dim() == 2:
                features = output
            else:
                features = output[:, 0, :]

        return features.cpu().numpy()[0]

    def draw_trajectory(self, frame: np.ndarray, track: TrackState, color: tuple):
        """Draw position history trail for a track."""
        if len(track.positions) < 2:
            return
        
        positions = list(track.positions)
        for i in range(1, len(positions)):
            # Fade older points
            alpha = i / len(positions)
            thickness = max(1, int(3 * alpha))
            
            pt1 = positions[i - 1]
            pt2 = positions[i]
            cv2.line(frame, pt1, pt2, color, thickness)

    def draw_reid_event(self, frame: np.ndarray, event: ReIDEvent):
        """Draw Re-ID match notification."""
        # Flash green banner at top
        text = f"RE-ID MATCH: Track {event.old_id} -> {event.new_id} ({event.similarity:.0%})"
        
        # Calculate alpha for fade-out
        alpha = event.frames_remaining / 30.0
        
        # Draw semi-transparent banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 50), (0, 200, 0), -1)
        cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)
        
        # Draw text
        cv2.putText(frame, text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)

    def draw_detection(self, frame: np.ndarray, detection: dict, 
                       track: TrackState):
        """Draw bounding box, label, and optional trajectory."""
        x1, y1, x2, y2 = detection['box']
        label = detection['label']
        track_id = detection['track_id']
        is_locked = detection['locked']
        
        color = TEAM_COLORS.get(label, (128, 128, 128))

        # Draw trajectory if enabled
        if self.args.draw_trajectory:
            self.draw_trajectory(frame, track, color)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        display_label = label.replace('_', ' ').upper()
        lock_icon = " [LOCKED]" if is_locked else ""
        label_text = f"#{track_id} {display_label}{lock_icon}"

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Get text size
        (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

        # Draw label background
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 8, y1), color, -1)

        # Draw label text
        cv2.putText(frame, label_text, (x1 + 4, y1 - 5), font, 
                   font_scale, (255, 255, 255), thickness)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame."""
        height, width = frame.shape[:2]

        # Run YOLO with ByteTrack
        results = self.yolo.track(
            frame,
            persist=True,
            classes=[self.args.target_class],
            conf=self.args.conf_thresh,
            verbose=False,
            tracker="bytetrack.yaml"
        )

        # Skip if no detections
        if results[0].boxes.id is None:
            self.manager.cleanup(set())
            
            # Still draw Re-ID events if any
            if self.args.draw_reid_events:
                for event in self.manager.reid_events:
                    self.draw_reid_event(frame, event)
            
            return frame

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        detections = []
        tracks_for_drawing = {}
        seen_ids = set()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)

            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Skip tiny detections
            if (x2 - x1) < self.args.min_box_size or (y2 - y1) < self.args.min_box_size:
                continue

            # Crop and convert BGR -> RGB
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Extract features
            features = self.extract_features(crop_rgb)
            
            # Calculate center position
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Get or create track (with Re-ID matching)
            track, reid_event = self.manager.get_or_create_track(
                track_id, features, center
            )
            seen_ids.add(track_id)
            tracks_for_drawing[track_id] = track

            # Classify and update track
            if track.locked_label:
                final_label = track.locked_label
                conf = 1.0
            else:
                probs = self.classifier.predict_proba(features.reshape(1, -1))[0]
                pred_idx = np.argmax(probs)
                raw_label = self.classes[pred_idx]
                conf = probs[pred_idx]

                final_label = track.update(raw_label, features, center)

            detections.append({
                'track_id': track_id,
                'box': (x1, y1, x2, y2),
                'label': final_label,
                'conf': conf,
                'locked': track.locked_label is not None
            })

        # Enforce team constraint
        detections = self.manager.enforce_team_constraint(detections)

        # Draw all detections
        for det in detections:
            track = tracks_for_drawing.get(det['track_id'])
            if track:
                self.draw_detection(frame, det, track)

        # Draw Re-ID events if enabled
        if self.args.draw_reid_events:
            for event in self.manager.reid_events:
                self.draw_reid_event(frame, event)

        # Cleanup stale tracks
        self.manager.cleanup(seen_ids)

        return frame

    def run(self):
        """Run the full pipeline on input video."""
        cap = cv2.VideoCapture(self.args.source)
        if not cap.isOpened():
            logging.error(f"❌ Failed to open video: {self.args.source}")
            return

        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logging.info(f"📹 Input: {self.args.source}")
        logging.info(f"   Resolution: {width}x{height} @ {fps:.1f}fps")
        logging.info(f"   Frames: {total_frames}")

        # Setup output
        os.makedirs(os.path.dirname(self.args.output), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.args.output, fourcc, fps, (width, height))

        logging.info(f"📼 Output: {self.args.output}")
        logging.info(f"🎬 Processing (stride={self.args.stride})...")

        frame_idx = 0
        processed_count = 0

        with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.args.stride == 0:
                    frame = self.process_frame(frame)
                    processed_count += 1

                out.write(frame)
                frame_idx += 1
                pbar.update(1)

        cap.release()
        out.release()

        # Log final metrics
        metrics = self.manager.get_metrics()
        logging.info("=" * 60)
        logging.info("📊 Re-ID Metrics:")
        logging.info(f"   Total Re-ID Matches: {metrics['total_reid_matches']}")
        logging.info(f"   Total New Tracks: {metrics['total_new_tracks']}")
        logging.info(f"   ID Switches Prevented: {metrics['potential_id_switches_prevented']}")
        logging.info("=" * 60)
        logging.info(f"✅ Complete! Processed {processed_count}/{total_frames} frames")
        logging.info(f"✅ Output: {self.args.output}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    args = parse_args()
    setup_logger(args.log)

    logging.info("=" * 60)
    logging.info("MotoGP Team Detection Pipeline v3")
    logging.info("Deep Re-ID & State Estimation for High-Velocity Agents")
    logging.info("=" * 60)

    pipeline = MotoGPPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
