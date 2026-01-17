# rPPG Service

Remote Photoplethysmography (rPPG) service for extracting heart rate and cardiovascular signals from video using facial analysis.

## Features

- **Primary Method**: PhysNet model from rPPG-Toolbox
- **Backup Method**: FFT-based analysis using green channel extraction
- Face detection using MediaPipe (primary) or OpenCV Haar Cascade (fallback)
- Heart rate (BPM) calculation
- HRV metrics: SDNN, RMSSD, pNN50
- Blood pressure estimation using PPG morphology
- Signal quality assessment (SNR-based)
- Comprehensive error handling and logging

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Pre-trained PhysNet Model (Optional but Recommended)

The service will work without pre-trained weights, but accuracy will be lower. To download:

```bash
# Option 1: Download from rPPG-Toolbox repository
# Visit: https://github.com/ubicomplab/rPPG-Toolbox
# Download PhysNet model weights and place in models/ directory

# Option 2: Set environment variable with model path
export PHYSNET_MODEL_PATH=/path/to/physnet_model.pth
```

### 3. Install Backup Library (Optional)

For enhanced backup processing:

```bash
pip install git+https://github.com/prouast/heartbeat.git
```

## Usage

### Start the Service

```bash
python app.py
```

Or with custom port:

```bash
PORT=8001 python app.py
```

### API Endpoints

#### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "service": "rppg",
  "primary_available": true,
  "backup_available": true
}
```

#### Analyze Video

```bash
POST /analyze-video
Content-Type: multipart/form-data

# Upload video file (mp4, avi, mov, mkv, webm)
```

Response:
```json
{
  "success": true,
  "heart_rate": 72.5,
  "hrv_sdnn": 45.2,
  "hrv_rmssd": 38.7,
  "hrv_pnn50": 12.5,
  "estimated_systolic_bp": 120.0,
  "estimated_diastolic_bp": 80.0,
  "signal_quality_score": 0.85,
  "method": "physnet",
  "fps": 30.0,
  "frames_processed": 900,
  "message": "Analysis completed successfully"
}
```

## Video Requirements

- **Format**: mp4, avi, mov, mkv, webm
- **Duration**: Minimum 10 seconds, optimal 30 seconds
- **Frame Rate**: 30 fps recommended
- **Resolution**: Minimum 640x480
- **Lighting**: Good, even lighting on face
- **Face Visibility**: Full face visible, minimal movement
- **Background**: Plain background preferred

## Error Handling

The service handles various error conditions:

- **Face not detected**: Returns 400 error with guidance
- **Video too short**: Returns 400 error (minimum 10 seconds)
- **Poor lighting**: Detected via low signal quality score
- **Invalid file format**: Returns 400 error with allowed formats
- **Processing failures**: Falls back to backup processor if available

## Signal Quality Assessment

Signal quality is scored from 0.0 to 1.0 based on:

- Signal-to-Noise Ratio (SNR)
- Signal stability across time windows
- Frequency domain analysis

**Quality Thresholds**:
- > 0.7: Excellent quality
- 0.5 - 0.7: Good quality
- 0.3 - 0.5: Acceptable quality
- < 0.3: Poor quality (warning issued)

## Architecture

### Primary Processor (PhysNet)

1. **Face Detection**: MediaPipe or OpenCV Haar Cascade
2. **ROI Extraction**: Face region extraction and resizing
3. **Signal Extraction**: PhysNet neural network
4. **Signal Processing**: Bandpass filtering (0.7-4 Hz)
5. **Analysis**: Heart rate, HRV, BP estimation

### Backup Processor (FFT-based)

1. **Face Detection**: OpenCV Haar Cascade
2. **Forehead ROI**: Extract forehead region (best for rPPG)
3. **Green Channel**: Extract mean green channel values
4. **FFT Analysis**: Frequency domain analysis
5. **Peak Detection**: Find heart rate from frequency peaks

## Model Integration

### Using Pre-trained PhysNet Model

1. Download model weights from rPPG-Toolbox:
   ```bash
   # Clone repository
   git clone https://github.com/ubicomplab/rPPG-Toolbox.git
   cd rPPG-Toolbox
   
   # Download PhysNet weights (check repository for latest links)
   # Place in models/ directory
   ```

2. Set environment variable:
   ```bash
   export PHYSNET_MODEL_PATH=/path/to/models/PhysNet.pth
   ```

3. Restart service

### Model Architecture

PhysNet is a 3D CNN that processes video clips to extract rPPG signals:
- Input: 160 frames × 128×128 RGB
- Output: rPPG signal time series
- Architecture: Encoder-decoder with temporal convolutions

## Logging

The service includes comprehensive logging:

- **INFO**: Normal operations, processing steps
- **WARNING**: Fallback processor usage, low quality signals
- **ERROR**: Processing failures, critical errors

Log format:
```
2024-01-13 10:30:45 - rppg_processor - INFO - Extracted 900 face frames
2024-01-13 10:30:46 - app - INFO - Analysis complete: HR=72.5, Quality=0.85
```

## Performance Considerations

- **GPU**: CUDA-enabled GPU recommended for PhysNet (10x faster)
- **CPU**: Works on CPU but slower
- **Memory**: ~2GB RAM for 30-second video processing
- **Processing Time**: 
  - GPU: ~5-10 seconds per video
  - CPU: ~30-60 seconds per video

## Troubleshooting

### Face Not Detected

- Ensure good lighting on face
- Face should be clearly visible
- Minimize head movement
- Use plain background

### Low Signal Quality

- Improve lighting conditions
- Reduce camera shake
- Ensure stable face position
- Use higher resolution video

### Model Loading Issues

- Check model file path
- Verify PyTorch installation
- Check CUDA availability (if using GPU)
- Service will fall back to FFT method if model unavailable

## References

- rPPG-Toolbox: https://github.com/ubicomplab/rPPG-Toolbox
- PhysNet Paper: "PhysNet: A Novel Framework for Remote Photoplethysmography"
- Heartbeat Library: https://github.com/prouast/heartbeat

## License

See main project LICENSE file.
