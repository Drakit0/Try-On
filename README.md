# Try-On

Webcam virtual dress try-on built as a side demo for Lady Pipa during the EY-Parthenon Race to Parthenon 2025. It is a Streamlit app that finds the body in the video, warps a dress image to the pose and draws it over the person in real time.

Authors: Pablo Tuñón Laguna and Lydia Ruiz Martínez.

## How it works

For each frame from the webcam (`app.py`, streamlit-webrtc):

1. `src/pose.py` runs MediaPipe pose detection and returns the body landmarks (shoulders, hips). It uses the MediaPipe Tasks pose landmarker when the `pose_landmarker_full.task` model file is available and the classic MediaPipe solutions API otherwise (`src/mediapipe_compat.py`). Pose runs on every second frame and the last result is reused in between.
2. `src/smoothing.py` applies a One Euro filter to each landmark to reduce jitter.
3. `src/overlay.py` and `src/tps_mapper.py` compute where the garment goes. The garment PNG has named control points (shoulder seams, waist, hips, hem) stored in `assets/garments.json`. The mapper turns the landmarks into destination points for those control points.
4. `src/tps.py` fits a thin-plate spline between the garment control points and the destination points and warps the garment image. With the warp turned off, the garment is only rotated, scaled and translated.
5. `src/segmenter.py` runs MediaPipe person segmentation. When occlusion is on, the parts of the person that should stay in front of the dress are kept on top of it.
6. The garment is alpha-blended onto the frame.

The sidebar has the garment selector, size and vertical offset sliders, toggles for the spline warp, occlusion, mirror view, FPS counter and debug statistics, and a capture button. Everything runs locally. A frame is saved only when the capture button is pressed, into a temporary folder, and "Delete All" removes the captures (`src/privacy.py`).

## Run

Python 3.9 or newer and a webcam.

```
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Press START in the video widget to turn the camera on. Browsers only allow camera access on localhost or over HTTPS. `launch_demo.py` checks the dependencies and assets and then starts the same app. On Linux hosts `packages.txt` lists the system libraries needed by OpenCV and ffmpeg.

`make.ps1` (Windows) and the `Makefile` wrap setup, run, test, lint and clean.

## Garments

`assets/garments/` has two dresses as JPG and as PNG with transparent background: Vestido Siracusa Burdeos and Vestido Nubia Marino. Both have six control points in `assets/garments.json`. `setup/download_dress.py` and `setup/remove_background.py` fetch the Siracusa photo and cut out the background (the second needs `rembg`). `setup_dresses.py` generates control points from the image size.

To add a garment, put a transparent PNG in `assets/garments/`, mark its control points with

```
streamlit run tools/point_picker.py
```

add the entry to `assets/garments.json`, and check it with `python tools/validate_garments.py` and `python tools/validate_control_points.py`.

## Tools and tests

- `tools/benchmark_fps.py`, `tools/profile_performance.py` and `tools/line_profiler_script.py` measure frame rate and time per stage.
- `verify_demo.py` checks that `garments.json` and the image files match.
- `tests/` has pytest tests for the modules in `src/`: `pytest`.

## Licence

MIT, see `LICENSE`. The dress photographs belong to Lady Pipa and are not covered by it.
