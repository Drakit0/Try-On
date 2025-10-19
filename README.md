# 🎭 Virtual Try-On Demo

![CI](https://github.com/Drakit0/virtual-tryon/actions/workflows/ci.yml/badge.svg)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-245%20passing-brightgreen.svg)](tests/)

**Real-time virtual garment try-on** using your webcam! Features pose detection, TPS warping, and realistic fitting. Try Lady Pipa dresses instantly.

## 🚀 Quick Start

```bash
# Launch the demo
python launch_demo.py
```

Or manually:
```bash
streamlit run app.py
```

**Opens at:** http://localhost:8501

> 📖 **Full demo guide:** See [DEMO_README.md](DEMO_README.md)

## 🌟 Lady Pipa Integration

Try **Vestido Siracusa Burdeos** from [Lady Pipa](https://ladypipa.com)!

### Quick Setup

```bash
# 1. Download the dress
python download_dress.py

# 2. Remove background (choose one):
# Option A: Automated (requires rembg)
pip install rembg
python remove_background.py

# Option B: Manual (use remove.bg website)
# Upload assets/garments/vestido_siracusa_burdeos.jpg to https://remove.bg
# Download PNG and save as assets/garments/vestido_siracusa_burdeos.png

# 3. Mark control points
streamlit run tools/point_picker.py

# 4. Launch demo
python launch_demo.py
```

See [DEMO_README.md](DEMO_README.md) for detailed instructions.

## ⚡ Features

- **Pose Landmarker**: Shoulders and hips detection for anchoring and scaling
- **Segmentation**: Skin/face/person segmentation for realistic occlusion
- **2D Thin-Plate Splines (TPS)**: Warp transparent PNG dresses to fit body contours
- **TPS Mapper**: Automatic control point computation from pose landmarks
- **Real-time Webcam**: Streamlit UI with webcam capture via streamlit-webrtc
- **Performance**: Target 20–30 FPS @ 640×480 (desktop), 15–24 FPS (mobile)

## Core Modules

### 1. Pose Detection (`pose.py`)
Detects body landmarks (shoulders, hips) using MediaPipe Pose Landmarker. Provides normalized coordinates for anchoring garments to the body.

### 2. Segmentation (`segmenter.py`)
Generates person/skin/face masks using MediaPipe Image Segmenter. Enables realistic occlusion (face/arms visible over garment).

### 3. TPS Warping (`tps.py`)
Implements 2D Thin-Plate Spline interpolation for non-rigid garment deformation. Warps transparent PNG garments using source and destination control points.

### 4. TPS Mapper (`tps_mapper.py`)
**NEW:** Computes destination control points from pose landmarks frame-by-frame.

**Key Features:**
- **Shoulder Mapping**: Direct mapping to pose shoulder landmarks
- **Waist Interpolation**: `lerp(shoulders, hips, alpha=0.6) + y_offset`
- **Lateral Adjustment**: `±(0.08 * shoulder_distance)` for waist width
- **Hip Widening**: Optional expansion `±(0.05 * shoulder_distance)` based on segmentation
- **Hem Extrapolation**: Vertical extension below hips for dresses
- **Pre-transform**: Optional rigid rotation/scale before TPS (reduces warp stress)
- **Real-time Tuning**: Adjustable parameters via UI for quick iteration

**Heuristics:**
```python
# Shoulders: Direct mapping
dst_l_shoulder = landmarks['left_shoulder']
dst_r_shoulder = landmarks['right_shoulder']

# Waist: Interpolation with offset
mid_waist = lerp(mid_shoulders, mid_hips, 0.6) + (0, y_offset_to_waist_px)
dst_l_waist = mid_waist + (-0.08 * shoulder_distance, 0)
dst_r_waist = mid_waist + (+0.08 * shoulder_distance, 0)

# Hips: With optional widening
dst_l_hip = landmarks['left_hip'] + (-widening, 0)
dst_r_hip = landmarks['right_hip'] + (+widening, 0)
```

### 5. Overlay & Compositing (`overlay.py`)
Blends warped garment with video frame using alpha compositing. Supports occlusion masks for realistic layering.

### 6. Temporal Smoothing (`smoothing.py`)
Reduces jitter in landmark positions using exponential moving average (EMA) and Kalman filtering.

## Project Structure

```
virtual_tryon_py/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── src/                    # Source modules
│   ├── pose.py            # Pose landmark detection
│   ├── segmenter.py       # Segmentation module
│   ├── overlay.py         # Overlay and compositing
│   ├── tps.py             # Thin-Plate Splines warping
│   ├── tps_mapper.py      # TPS control point mapping
│   ├── smoothing.py       # Temporal smoothing
│   └── utils.py           # Utility functions
├── assets/                 # Asset files
│   ├── garments/          # Transparent PNG dresses
│   └── garments.json      # Garment metadata
├── tests/                  # Unit tests
├── docs/                   # Documentation
│   ├── ops_guide.md       # Asset preparation guide
│   └── png_requirements.md # PNG quality standards
├── tools/                  # Validation tools
│   └── validate_garments.py
└── README.md              # This file
```

## Quick Setup

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Using Phone Camera

To use your phone's camera with the application, you need to serve the app over HTTPS:

### Option 1: ngrok (Easiest)
```bash
# Install ngrok from https://ngrok.com/
ngrok http 8501
```
Use the HTTPS URL provided by ngrok to access the app from your phone.

### Option 2: Caddy
```bash
# Install Caddy from https://caddyserver.com/
caddy reverse-proxy --from :443 --to :8501
```

### Option 3: Custom Reverse Proxy
Configure your reverse proxy (nginx, Apache, etc.) to forward HTTPS traffic to `http://localhost:8501`.

**Note:** Most modern browsers require HTTPS to access device cameras for security reasons.

For detailed HTTPS setup instructions, see [`scripts/dev_https.md`](scripts/dev_https.md).

## Development

### Using Build Scripts

The project includes build automation for common tasks:

**PowerShell (Windows - Recommended):**
```powershell
.\make.ps1 help      # Show all available commands
.\make.ps1 setup     # Create venv and install dependencies
.\make.ps1 run       # Run the Streamlit app
.\make.ps1 test      # Run tests
.\make.ps1 lint      # Run code linting
.\make.ps1 format    # Format code with black
.\make.ps1 clean     # Clean generated files
.\make.ps1 dev       # Install dev dependencies
```

**Makefile (Linux/macOS/Windows with Make):**
```bash
make help            # Show all available commands
make setup           # Create venv and install dependencies
make run             # Run the Streamlit app
make test            # Run tests
make lint            # Run code linting
make format          # Format code with black
make clean           # Clean generated files
make dev             # Install dev dependencies
```

### Running Tests Manually

Run all tests:
```bash
pytest -q
```

Run specific test suites:
```bash
# TPS tests (warping accuracy, alpha preservation)
pytest tests/test_tps.py -v

# Transform tests (angle computation, scale coherence)
pytest tests/test_overlay.py -v

# Segmentation tests (mask size, binary values, border smoothing)
pytest tests/test_segmenter.py -v
```

Or with the build script:
```powershell
.\make.ps1 test
```

**Test Coverage:**
- `tests/test_overlay.py`: 43 tests - Transform computation, overlay rendering
- `tests/test_tps.py`: 33 tests - TPS warping, control points, performance
- `tests/test_smoothing.py`: 29 tests - Temporal filtering, jitter reduction
- `tests/test_tps_mapper.py`: 28 tests - TPS control point mapping from pose
- `tests/test_segmenter.py`: 28 tests - Mask generation, refinement, occlusion
- `tests/test_privacy.py`: 27 tests - Capture management, privacy safeguards
- `tests/test_pose.py`: 20 tests - Pose detection, landmark extraction
- `tests/test_placeholder.py`: 1 test - Basic setup verification

**Total: 209 tests**

### Project Conventions

- **Data types**: `np.ndarray` BGR for frames; RGBA for garments
- **Logging**: Standard Python `logging` module
- **Style**: Google/NumPy docstrings with `typing` annotations
- **Privacy**: All processing is done locally; no data is sent to external servers

### MediaPipe Compatibility

The project supports both MediaPipe Tasks API (recommended) and classic MediaPipe Solutions API as a fallback. The compatibility layer automatically detects available APIs:

```bash
# Check MediaPipe availability
python src/mediapipe_compat.py
```

Output will show:
- MediaPipe version
- Tasks API availability
- Recommended detector types

The codebase includes `# Fallback` comments where classic Solutions API is used when Tasks API is unavailable.

## Performance Tips

- Use lower resolution (640×480) for better frame rates
- Enable frame skipping for resource-constrained devices
- Precompute TPS transformations where possible
- Use GPU acceleration if available

## Version Pinning

For reproducible builds, exact package versions are tracked in `docs/pip_freeze_example.txt`. This file contains the output of `pip freeze` from a working development environment.

**Note:** This file is for reference only and should not be used directly in production. Use `requirements.txt` for installation, which specifies version ranges for flexibility.

To regenerate this file in your environment:
```bash
pip freeze > docs/pip_freeze_example.txt
```

Key dependencies:
- Python: 3.8+
- Streamlit: 1.x
- MediaPipe: ≥0.10
- OpenCV: Latest stable
- NumPy, SciPy, scikit-image: Latest compatible versions

## Privacy & Security

### Local Processing Only

🔒 **All AI processing happens locally on your device.** No video frames, pose data, or biometric information is sent to external servers.

### Data Storage Policy

- ✅ **No persistent storage**: Images are **not stored** unless you explicitly press the "Capture" button
- ✅ **Temporary captures**: Saved to system temp directory (typically cleared by OS automatically)
- ✅ **User control**: Delete all captures anytime via "Delete All" button in the UI
- ✅ **No biometric persistence**: Pose landmarks computed per-frame and immediately discarded
- ✅ **No analytics**: No tracking, cookies, or external service calls

### HTTPS Requirement

⚠️ **Mobile camera access requires HTTPS.** Modern browsers require secure connections for camera access on mobile devices.

Use one of these methods for HTTPS:

**Option 1: ngrok (Easiest)**
```bash
streamlit run app.py &
ngrok http 8501
```

**Option 2: Caddy**
```bash
caddy reverse-proxy --from :443 --to :8501
```

**Option 3: Custom Reverse Proxy**
Configure nginx, Apache, or other reverse proxy to forward HTTPS to `http://localhost:8501`.

See [`scripts/dev_https.md`](scripts/dev_https.md) for detailed HTTPS setup.

### Capture Management

**Capture Location**: Captures saved to `{system_temp}/virtual_tryon_captures/`

**Capture Controls**:
- 📷 **Capture Button**: Save current frame with metadata (garment name, settings, timestamp)
- 🗑️ **Delete All Button**: Remove all captures from local storage
- 📊 **Storage Stats**: View number of captures and total storage size

**Privacy Notice in UI**:
> 🔒 **Local processing only** • Images not stored unless you press Capture

## Limitations

### Technical Limitations

- **2D Warping Only**: No 3D modeling or depth awareness; garments warped in 2D plane
- **Single Garment Layer**: Cannot overlay multiple garments simultaneously
- **Pose Dependency**: Requires clear body pose detection (well-lit, front-facing, unobstructed)
- **Garment Types**: Optimized for shirts/dresses; complex garments (jackets, skirts) may not warp realistically
- **Performance**: Frame rate varies by device (15-30 FPS typical; 10-15 FPS on mobile)

### Privacy & Security Limitations

- **Local Processing Required**: Needs sufficient local compute (CPU/GPU); no cloud processing available
- **No Cloud Backup**: Captures stored locally only; lost if temp directory cleared by OS
- **Camera Permission Required**: Browser must grant camera access (HTTPS required on mobile)
- **No Encryption**: Captures stored as plain PNG files in temp directory (not encrypted at rest)
- **Session Data**: Pose landmarks kept in memory during session for smoothing (cleared on page refresh)

### Functional Limitations

- **Lighting Sensitivity**: Performance degrades in low light, harsh shadows, or high contrast
- **Occlusion Handling**: Garment may not handle complex arm positions or bent postures correctly
- **Scale Range**: Works best for upper body garments (shirts, dresses, blouses); full-body outfits not supported
- **Transparency**: Very sheer/translucent garments may appear incorrect due to alpha blending
- **Pattern Distortion**: Complex patterns may visibly distort during TPS warping
- **Body Type Variance**: Best results with average body proportions matching garment design

### Browser Compatibility

- **WebRTC Required**: Modern browsers only (Chrome 60+, Firefox 55+, Edge 79+, Safari 11+)
- **HTTPS Required**: Mobile camera access requires HTTPS connection
- **GPU Acceleration**: Limited browser GPU support for MediaPipe (CPU fallback may be slower)
- **iOS Limitations**: Safari on iOS may have reduced performance compared to desktop

### Known Issues

- **Shoulder Alignment**: May misalign if user's shoulders not parallel to camera
- **Waist Position**: Fixed interpolation may not match all body types (adjustable via TPS mapper params)
- **Hem Behavior**: Long dresses may appear disconnected from body at bottom edge
- **Segmentation Artifacts**: Person segmentation may include/exclude incorrect regions in cluttered backgrounds

## License

See LICENSE file for details.

## Credits

Built with:
- [Streamlit](https://streamlit.io/)
- [MediaPipe](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)

## Documentation

### Technical Documentation

Comprehensive technical documentation for developers:

- 📐 **[System Architecture](docs/architecture.md)** - High-level architecture, module responsibilities, data pipeline, performance architecture
- 🔄 **[Data Flow](docs/data_flow.md)** - Data types, coordinate systems, color spaces, module contracts, video loop events
- 📚 **[API Reference](docs/api_reference.md)** - Complete API documentation with signatures, parameters, returns, exceptions, usage examples
- 🧮 **[TPS Mathematics](docs/tps_math.md)** - Thin-Plate Spline mathematical foundation, L matrix construction, numerical considerations
- ⚙️ **[Operations Guide](docs/ops_guide.md)** - Getting started, HTTPS setup, performance tuning, troubleshooting, asset management

### User Guides

- 📋 **[Quick Reference Card](docs/quick_reference.md)** - Common issues and instant solutions
- 📖 **[Full Troubleshooting Guide](docs/troubleshooting.md)** - Detailed problem-solving
- 🔒 **[Privacy Guide](docs/privacy_guide.md)** - Privacy and security information

### Stakeholder Materials

- 🎤 **[Pitch Deck](docs/pitch_deck.md)** - Business case, ROI analysis, competitive landscape, roadmap
- 🎬 **[Demo Script](docs/demo_script_90s.md)** - 90-second demo walkthrough with Q&A prep

## Troubleshooting

**Most Common Issues:**

- **Camera not working on mobile?** → Use HTTPS (see [Quick Ref](docs/quick_reference.md#-camera-not-working-on-mobile) or [Ops Guide](docs/ops_guide.md#https-setup))
- **Low FPS?** → Reduce resolution or disable TPS (see [Performance](docs/troubleshooting.md#-performance-issues) or [Ops Guide](docs/ops_guide.md#performance-tuning))
- **Edge artifacts?** → Increase `eps` in TPS (see [Visual Quality](docs/troubleshooting.md#-visual-quality-issues) or [TPS Math](docs/tps_math.md#numerical-considerations))
- **Installation problems?** → Check [Dependencies](docs/troubleshooting.md#-installation--dependencies) or [Ops Guide](docs/ops_guide.md#installation)

