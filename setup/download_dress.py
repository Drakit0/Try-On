"""
Download and prepare Vestido Siracusa Burdeos from Lady Pipa
"""
import urllib.request
from pathlib import Path

# Dress image URL from Lady Pipa
DRESS_URL = "https://ladypipa.com/cdn/shop/files/LADYPIPARUNWAY8920.jpg?v=1754907090&width=1000"
OUTPUT_FILE = Path("assets/garments/vestido_siracusa_burdeos.jpg")

print("Downloading Vestido Siracusa Burdeos...")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

try:
    urllib.request.urlretrieve(DRESS_URL, OUTPUT_FILE)
    print(f"✅ Downloaded to: {OUTPUT_FILE}")
    print(f"   File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    print("\nNext steps:")
    print("1. Remove background from the image (use remove.bg or Photoshop)")
    print("2. Save as PNG with transparent background")
    print("3. Use point picker tool to mark control points:")
    print("   streamlit run tools/point_picker.py")
except Exception as e:
    print(f"❌ Error downloading: {e}")
