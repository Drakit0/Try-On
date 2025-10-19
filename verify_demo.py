"""
Verification script for Lady Pipa Virtual Try-On Demo
Checks that all components are properly configured
"""
import json
from pathlib import Path
from PIL import Image

print("=" * 70)
print("🔍 LADY PIPA DEMO VERIFICATION")
print("=" * 70)
print()

# Check 1: Garments JSON
print("1. Checking garments.json...")
garments_file = Path("assets/garments.json")
if garments_file.exists():
    with open(garments_file) as f:
        data = json.load(f)
    garments = data.get("garments", [])
    print(f"   ✓ Found {len(garments)} garments")
    for g in garments:
        print(f"     - {g['name']} ({g['id']})")
else:
    print("   ❌ garments.json not found!")

print()

# Check 2: PNG Images
print("2. Checking PNG images...")
required_images = [
    "assets/garments/vestido_siracusa_burdeos.png",
    "assets/garments/vestido_nubia_marino.png"
]
for img_path in required_images:
    if Path(img_path).exists():
        img = Image.open(img_path)
        print(f"   ✓ {Path(img_path).name}")
        print(f"     Size: {img.size}, Mode: {img.mode}, Has alpha: {img.mode == 'RGBA'}")
    else:
        print(f"   ❌ {img_path} not found!")

print()

# Check 3: Control Points
print("3. Checking control points...")
for g in garments:
    points = g.get("tps_control_points", {}).get("src", [])
    print(f"   {g['name']}: {len(points)} points")
    if len(points) == 6:
        print(f"     ✓ Correct number of control points")
    else:
        print(f"     ❌ Expected 6 points, got {len(points)}")

print()

# Check 4: Dependencies
print("4. Checking dependencies...")
dependencies = [
    ("streamlit", "Streamlit"),
    ("streamlit_webrtc", "Streamlit WebRTC"),
    ("cv2", "OpenCV"),
    ("mediapipe", "MediaPipe"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("PIL", "Pillow"),
    ("skimage", "Scikit-image"),
]

all_ok = True
for module, name in dependencies:
    try:
        __import__(module)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ❌ {name} - NOT INSTALLED")
        all_ok = False

print()

# Check 5: App imports
print("5. Checking app imports...")
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from src import pose, segmenter, tps, overlay, privacy, utils
    print("   ✓ All source modules import successfully")
except Exception as e:
    print(f"   ❌ Import error: {e}")
    all_ok = False

print()

# Final verdict
print("=" * 70)
if all_ok:
    print("✅ ALL CHECKS PASSED!")
    print()
    print("Your demo is ready to run!")
    print()
    print("Start the demo with:")
    print("  python launch_demo.py")
    print()
    print("Or directly:")
    print("  streamlit run app.py")
else:
    print("⚠️  SOME CHECKS FAILED")
    print()
    print("Please resolve the issues above before running the demo.")
print("=" * 70)
