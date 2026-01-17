"""
rPPG Signal Processor using PhysNet from rPPG-Toolbox
Handles video processing, face detection, and signal extraction
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import logging
import os
from typing import Tuple, Optional, Dict, List
from scipy import signal
from scipy.stats import pearsonr
import mediapipe as mp

logger = logging.getLogger(__name__)


class PhysNet(nn.Module):
    """
    PhysNet model architecture for rPPG signal extraction
    Based on: https://github.com/ubicomplab/rPPG-Toolbox
    """
    
    def __init__(self, frames=160):
        super(PhysNet, self).__init__()
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 32, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 64, [1, 4, 4], stride=[1, 2, 2], padding=[0, 1, 1]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, [1, 4, 4], stride=[1, 2, 2], padding=[0, 1, 1]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        
    def forward(self, x):
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.upsample(x)
        x = self.upsample2(x)
        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.ConvBlock8(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1), -1)
        return x


class FaceDetector:
    """Face detection using MediaPipe (primary) or OpenCV Haar Cascade (fallback)"""
    
    def __init__(self):
        self.mediapipe_available = False
        self.haar_available = False
        
        # Try MediaPipe first
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # Full range model
                min_detection_confidence=0.5
            )
            self.mediapipe_available = True
            logger.info("MediaPipe face detection initialized")
        except Exception as e:
            logger.warning(f"MediaPipe initialization failed: {e}")
        
        # Fallback to Haar Cascade
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                self.haar_available = True
                logger.info("OpenCV Haar Cascade face detection initialized")
            else:
                logger.warning("Haar Cascade XML file not found")
        except Exception as e:
            logger.warning(f"Haar Cascade initialization failed: {e}")
        
        if not self.mediapipe_available and not self.haar_available:
            raise RuntimeError("No face detection method available")
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in frame
        
        Args:
            frame: BGR image frame
            
        Returns:
            Tuple of (x, y, width, height) or None if no face detected
        """
        # Try MediaPipe first
        if self.mediapipe_available:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_frame)
                
                if results.detections:
                    detection = results.detections[0]  # Use first detection
                    bbox = detection.location_data.relative_bounding_box
                    h, w = frame.shape[:2]
                    
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    return (x, y, width, height)
            except Exception as e:
                logger.warning(f"MediaPipe detection failed: {e}")
        
        # Fallback to Haar Cascade
        if self.haar_available:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    return (x, y, w, h)
            except Exception as e:
                logger.warning(f"Haar Cascade detection failed: {e}")
        
        return None


class RPPGProcessor:
    """Main rPPG signal processor using PhysNet"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize rPPG processor
        
        Args:
            model_path: Path to pre-trained PhysNet model weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize PhysNet model
        self.model = PhysNet(frames=160)
        self.model.to(self.device)
        self.model.eval()
        
        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model'])
                logger.info(f"Loaded model weights from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model weights: {e}. Using untrained model.")
        else:
            logger.warning("No model weights provided. Using untrained model.")
        
        # Initialize face detector
        self.face_detector = FaceDetector()
        
        # Processing parameters
        self.target_fps = 30
        self.clip_duration = 30  # seconds
        self.frame_size = (128, 128)
    
    def extract_face_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face region of interest
        
        Args:
            frame: Input frame
            
        Returns:
            Resized face ROI or None
        """
        face_bbox = self.face_detector.detect_face(frame)
        if face_bbox is None:
            return None
        
        x, y, w, h = face_bbox
        
        # Expand ROI slightly
        expand_factor = 0.2
        x = max(0, int(x - w * expand_factor))
        y = max(0, int(y - h * expand_factor))
        w = min(frame.shape[1] - x, int(w * (1 + 2 * expand_factor)))
        h = min(frame.shape[0] - y, int(h * (1 + 2 * expand_factor)))
        
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return None
        
        # Resize to target size
        face_roi = cv2.resize(face_roi, self.frame_size)
        
        return face_roi
    
    def preprocess_video(self, video_path: str) -> Tuple[np.ndarray, float]:
        """
        Preprocess video: extract frames and face ROIs
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (processed frames array, actual fps)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s duration")
        
        if duration < 10:
            cap.release()
            raise ValueError(f"Video too short: {duration:.2f}s. Minimum 10 seconds required.")
        
        # Extract frames
        frames = []
        frame_count = 0
        target_frame_count = int(self.clip_duration * self.target_fps)
        
        while cap.isOpened() and frame_count < target_frame_count:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract face ROI
            face_roi = self.extract_face_roi(frame)
            if face_roi is None:
                logger.warning(f"Face not detected in frame {frame_count}")
                continue
            
            frames.append(face_roi)
            frame_count += 1
        
        cap.release()
        
        if len(frames) < 30:  # Minimum frames for analysis
            raise ValueError(f"Insufficient frames with detected faces: {len(frames)}")
        
        frames_array = np.array(frames)
        logger.info(f"Extracted {len(frames)} face frames")
        
        return frames_array, fps
    
    def extract_rppg_signal(self, frames: np.ndarray) -> np.ndarray:
        """
        Extract rPPG signal using PhysNet
        
        Args:
            frames: Array of face ROI frames (N, H, W, C)
            
        Returns:
            rPPG signal array
        """
        # Prepare input tensor
        # PhysNet expects input shape: (batch, channels, frames, height, width)
        frames_tensor = torch.from_numpy(frames).float()
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (N, C, H, W)
        frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension
        frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4)  # (batch, frames, channels, H, W)
        
        # Normalize to [0, 1]
        frames_tensor = frames_tensor / 255.0
        
        # Move to device
        frames_tensor = frames_tensor.to(self.device)
        
        # Extract signal
        with torch.no_grad():
            output = self.model(frames_tensor)
            rppg_signal = output.cpu().numpy().flatten()
        
        return rppg_signal
    
    def calculate_heart_rate(self, rppg_signal: np.ndarray, fps: float) -> float:
        """
        Calculate heart rate from rPPG signal
        
        Args:
            rppg_signal: Extracted rPPG signal
            fps: Frames per second
            
        Returns:
            Heart rate in BPM
        """
        # Apply bandpass filter (0.7-4 Hz for heart rate)
        nyquist = fps / 2
        low = 0.7 / nyquist
        high = 4.0 / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, rppg_signal)
        
        # Find peaks
        peaks, _ = signal.find_peaks(filtered_signal, distance=int(fps * 0.5))
        
        if len(peaks) < 2:
            # Fallback: use FFT
            fft = np.fft.rfft(filtered_signal)
            freqs = np.fft.rfftfreq(len(filtered_signal), 1/fps)
            power = np.abs(fft) ** 2
            
            # Find peak in heart rate range (0.7-4 Hz)
            hr_range = (freqs >= 0.7) & (freqs <= 4.0)
            if np.any(hr_range):
                peak_freq = freqs[hr_range][np.argmax(power[hr_range])]
                hr = peak_freq * 60
            else:
                hr = 72.0  # Default fallback
        else:
            # Calculate HR from peak intervals
            intervals = np.diff(peaks) / fps
            hr = 60.0 / np.mean(intervals)
        
        return max(40, min(200, hr))  # Clamp to reasonable range
    
    def calculate_hrv(self, rppg_signal: np.ndarray, fps: float) -> Dict[str, float]:
        """
        Calculate HRV metrics: SDNN, RMSSD, pNN50
        
        Args:
            rppg_signal: Extracted rPPG signal
            fps: Frames per second
            
        Returns:
            Dictionary with HRV metrics
        """
        # Find peaks
        filtered_signal = signal.filtfilt(
            *signal.butter(4, [0.7/(fps/2), 4.0/(fps/2)], btype='band'),
            rppg_signal
        )
        peaks, _ = signal.find_peaks(filtered_signal, distance=int(fps * 0.5))
        
        if len(peaks) < 10:
            return {
                'sdnn': None,
                'rmssd': None,
                'pnn50': None
            }
        
        # Calculate RR intervals (in seconds)
        rr_intervals = np.diff(peaks) / fps
        
        # SDNN: Standard deviation of RR intervals
        sdnn = np.std(rr_intervals) * 1000  # Convert to ms
        
        # RMSSD: Root mean square of successive differences
        if len(rr_intervals) > 1:
            diff = np.diff(rr_intervals)
            rmssd = np.sqrt(np.mean(diff ** 2)) * 1000  # Convert to ms
        else:
            rmssd = None
        
        # pNN50: Percentage of intervals differing by more than 50ms
        if len(rr_intervals) > 1:
            diff_ms = np.abs(np.diff(rr_intervals)) * 1000
            pnn50 = (np.sum(diff_ms > 50) / len(diff_ms)) * 100
        else:
            pnn50 = None
        
        return {
            'sdnn': float(sdnn) if sdnn is not None else None,
            'rmssd': float(rmssd) if rmssd is not None else None,
            'pnn50': float(pnn50) if pnn50 is not None else None
        }
    
    def estimate_blood_pressure(self, rppg_signal: np.ndarray, fps: float) -> Dict[str, float]:
        """
        Estimate blood pressure using PPG morphology and pulse transit time
        
        Args:
            rppg_signal: Extracted rPPG signal
            fps: Frames per second
            
        Returns:
            Dictionary with estimated systolic and diastolic BP
        """
        # This is a simplified estimation - in production, use trained models
        filtered_signal = signal.filtfilt(
            *signal.butter(4, [0.7/(fps/2), 4.0/(fps/2)], btype='band'),
            rppg_signal
        )
        
        # Find peaks and valleys
        peaks, _ = signal.find_peaks(filtered_signal, distance=int(fps * 0.5))
        valleys, _ = signal.find_peaks(-filtered_signal, distance=int(fps * 0.5))
        
        if len(peaks) < 3 or len(valleys) < 3:
            # Default values if insufficient data
            return {
                'systolic': 120.0,
                'diastolic': 80.0
            }
        
        # Calculate pulse wave characteristics
        pulse_amplitudes = filtered_signal[peaks] - filtered_signal[valleys[:len(peaks)]]
        avg_amplitude = np.mean(pulse_amplitudes)
        
        # Simplified BP estimation (calibration needed for production)
        # These are rough estimates and should be calibrated with actual BP measurements
        systolic = 100 + (avg_amplitude * 50)
        diastolic = 60 + (avg_amplitude * 30)
        
        # Clamp to reasonable ranges
        systolic = max(90, min(180, systolic))
        diastolic = max(60, min(120, diastolic))
        
        return {
            'systolic': float(systolic),
            'diastolic': float(diastolic)
        }
    
    def assess_signal_quality(self, rppg_signal: np.ndarray, fps: float) -> float:
        """
        Assess signal quality using SNR and other metrics
        
        Args:
            rppg_signal: Extracted rPPG signal
            fps: Frames per second
            
        Returns:
            Quality score between 0 and 1
        """
        if len(rppg_signal) < 30:
            return 0.0
        
        # Calculate SNR
        signal_power = np.var(rppg_signal)
        
        # Estimate noise power (high frequency components)
        fft = np.fft.rfft(rppg_signal)
        freqs = np.fft.rfftfreq(len(rppg_signal), 1/fps)
        
        # Signal power in heart rate band (0.7-4 Hz)
        hr_band = (freqs >= 0.7) & (freqs <= 4.0)
        signal_band_power = np.sum(np.abs(fft[hr_band]) ** 2)
        
        # Noise power (outside heart rate band)
        noise_band_power = np.sum(np.abs(fft[~hr_band]) ** 2)
        
        if noise_band_power == 0:
            snr = 100  # Very high SNR
        else:
            snr = 10 * np.log10(signal_band_power / noise_band_power)
        
        # Normalize SNR to 0-1 scale (assuming good SNR is > 10 dB)
        quality_score = min(1.0, max(0.0, (snr + 10) / 30))
        
        # Additional quality checks
        # Check for signal stability (low variance in signal power)
        window_size = int(fps * 2)  # 2-second windows
        if len(rppg_signal) > window_size:
            windows = [rppg_signal[i:i+window_size] for i in range(0, len(rppg_signal)-window_size, window_size)]
            window_powers = [np.var(w) for w in windows]
            stability = 1.0 / (1.0 + np.std(window_powers) / np.mean(window_powers))
            quality_score = (quality_score + stability) / 2
        
        return float(quality_score)
    
    def process_video(self, video_path: str) -> Dict:
        """
        Complete video processing pipeline
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with all analysis results
        """
        try:
            # Preprocess video
            frames, fps = self.preprocess_video(video_path)
            
            # Extract rPPG signal
            rppg_signal = self.extract_rppg_signal(frames)
            
            # Calculate metrics
            heart_rate = self.calculate_heart_rate(rppg_signal, fps)
            hrv_metrics = self.calculate_hrv(rppg_signal, fps)
            bp_estimate = self.estimate_blood_pressure(rppg_signal, fps)
            quality_score = self.assess_signal_quality(rppg_signal, fps)
            
            return {
                'success': True,
                'heart_rate': heart_rate,
                'hrv_sdnn': hrv_metrics['sdnn'],
                'hrv_rmssd': hrv_metrics['rmssd'],
                'hrv_pnn50': hrv_metrics['pnn50'],
                'estimated_systolic_bp': bp_estimate['systolic'],
                'estimated_diastolic_bp': bp_estimate['diastolic'],
                'signal_quality_score': quality_score,
                'fps': fps,
                'frames_processed': len(frames)
            }
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise
