"""
🎭 Virtual Try-On Demo Launcher
================================

Launch the Virtual Try-On application with your webcam.
"""

import subprocess
import sys
from pathlib import Path

def print_banner():
    """Print demo banner."""
    print("=" * 70)
    print("🎭 LADY PIPA VIRTUAL TRY-ON DEMO")
    print("=" * 70)
    print("\n� Try on Lady Pipa dresses in real-time with your webcam!")
    print("   - Vestido Siracusa Burdeos")
    print("   - Vestido Nubia Marino\n")

def check_requirements():
    """Check if requirements are installed."""
    print("✓ Checking requirements...")
    try:
        import streamlit
        import cv2
        import mediapipe
        import numpy
        import PIL
        import scipy
        import skimage
        print("✓ All requirements installed\n")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements.txt\n")
        return False

def check_assets():
    """Check if garment assets exist."""
    print("✓ Checking assets...")
    garments_json = Path("assets/garments.json")
    
    if not garments_json.exists():
        print("❌ Missing assets/garments.json")
        return False
    
    print("✓ Assets found\n")
    return True

def print_instructions():
    """Print usage instructions."""
    print("=" * 70)
    print("📋 HOW TO USE THE DEMO")
    print("=" * 70)
    print("""
1. 📹 CAMERA: Click START button to activate your webcam
2. 👗 SELECT DRESS: Choose from sidebar dropdown:
   - Vestido Siracusa Burdeos (burgundy)
   - Vestido Nubia Marino (navy blue)
3. 🧍 POSITION: Stand 1-2 meters from camera, face forward
4. 🎚️  ADJUST FIT: Use sliders in sidebar:
   - Scale X/Y: Dress size
   - Rotation: Angle adjustment
   - Y Offset: Vertical position
5. 📷 CAPTURE: Take photos with "📷 Capture Frame" button

💡 TIPS: 
   - Good lighting is essential
   - Keep shoulders visible in frame
   - Stand against a plain background
   - Enable "Mirror Mode" if needed
""")

def launch_app():
    """Launch the Streamlit app."""
    print("🚀 Launching demo...\n")
    print("=" * 70)
    print("The app will open in your browser at:")
    print("👉 http://localhost:8501")
    print("=" * 70)
    print("\n✨ DEMO READY! The browser should open automatically...")
    print("   If not, manually open: http://localhost:8501")
    print("\nPress Ctrl+C to stop the demo\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped. Thanks for trying Lady Pipa Virtual Try-On!")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        print("\nTry running manually:")
        print("  streamlit run app.py")

def main():
    """Main demo launcher."""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check assets
    if not check_assets():
        print("\nPlease ensure assets/garments.json exists.")
        sys.exit(1)
    
    # Print instructions
    print_instructions()
    
    # Launch
    input("Press Enter to launch the demo... ")
    print()
    launch_app()

if __name__ == "__main__":
    main()
