"""
Remove background from Vestido Siracusa image
==============================================

Converts JPG to PNG with transparent background.
"""

try:
    from rembg import remove
    from PIL import Image
    from pathlib import Path
    
    print("🎨 Removing background from Vestido Siracusa...")
    
    input_path = Path("assets/garments/vestido_siracusa_burdeos.jpg")
    output_path = Path("assets/garments/vestido_siracusa_burdeos.png")
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        print("   Run download_dress.py first")
        exit(1)
    
    # Load image
    print(f"📂 Loading: {input_path}")
    img = Image.open(input_path)
    
    # Remove background
    print("✂️  Removing background (this may take 10-30 seconds)...")
    output = remove(img)
    
    # Save
    output.save(output_path)
    print(f"✅ Saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    print("\n📍 Next step:")
    print("   Mark control points using:")
    print("   streamlit run tools/point_picker.py")
    
except ImportError:
    print("❌ rembg not installed")
    print("\nInstall with:")
    print("  pip install rembg")
    print("\nAlternatives:")
    print("  1. Use remove.bg website (free, easy)")
    print("  2. Use Photoshop/GIMP (manual)")
    print("  3. Use online background remover")
