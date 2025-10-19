"""
Setup script for Lady Pipa dresses with automatic control points
"""
from PIL import Image
import json

def get_dress_control_points(image_path, dress_type="dress"):
    """Generate control points based on image dimensions."""
    img = Image.open(image_path)
    width, height = img.size
    
    # For dresses, control points are approximately:
    # Shoulders: 20% from sides, 10% from top
    # Waist: 25% from sides, 35% from top
    # Hips: 30% from sides, 55% from top
    
    points = [
        {
            "name": "L_shoulder_seam",
            "x": int(width * 0.20),
            "y": int(height * 0.10)
        },
        {
            "name": "R_shoulder_seam",
            "x": int(width * 0.80),
            "y": int(height * 0.10)
        },
        {
            "name": "L_waist",
            "x": int(width * 0.25),
            "y": int(height * 0.35)
        },
        {
            "name": "R_waist",
            "x": int(width * 0.75),
            "y": int(height * 0.35)
        },
        {
            "name": "L_hip",
            "x": int(width * 0.30),
            "y": int(height * 0.55)
        },
        {
            "name": "R_hip",
            "x": int(width * 0.70),
            "y": int(height * 0.55)
        }
    ]
    
    return points

# Setup Vestido Siracusa Burdeos
print("Setting up Vestido Siracusa Burdeos...")
img_siracusa = Image.open("assets/garments/vestido_siracusa_burdeos.png")
width_s, height_s = img_siracusa.size
print(f"  Image size: {width_s}x{height_s}")

siracusa_points = get_dress_control_points("assets/garments/vestido_siracusa_burdeos.png")
print(f"  Control points: {len(siracusa_points)}")

# Setup Vestido Nubia Marino
print("\nSetting up Vestido Nubia Marino...")
img_nubia = Image.open("assets/garments/vestido_nubia_marino.png")
width_n, height_n = img_nubia.size
print(f"  Image size: {width_n}x{height_n}")

nubia_points = get_dress_control_points("assets/garments/vestido_nubia_marino.png")
print(f"  Control points: {len(nubia_points)}")

# Create garments.json with both dresses
garments_data = {
    "garments": [
        {
            "id": "vestido_siracusa",
            "name": "Vestido Siracusa Burdeos",
            "path": "garments/vestido_siracusa_burdeos.png",
            "type": "dress",
            "metadata": {
                "base_shoulder_px": 250,
                "base_torso_px": 500,
                "y_offset_to_waist_px": 0
            },
            "tps_control_points": {
                "src": siracusa_points,
                "description": "Lady Pipa elegant dress - Control points for TPS warping"
            }
        },
        {
            "id": "vestido_nubia",
            "name": "Vestido Nubia Marino",
            "path": "garments/vestido_nubia_marino.png",
            "type": "dress",
            "metadata": {
                "base_shoulder_px": 240,
                "base_torso_px": 520,
                "y_offset_to_waist_px": -10
            },
            "tps_control_points": {
                "src": nubia_points,
                "description": "Lady Pipa elegant dress - Control points for TPS warping"
            }
        }
    ]
}

# Save to garments.json
with open("assets/garments.json", "w") as f:
    json.dump(garments_data, f, indent=2)

print("\n✓ Updated assets/garments.json with both dresses")
print("\nGarments configured:")
print("  1. Vestido Siracusa Burdeos")
print("  2. Vestido Nubia Marino")
print("\nYou can now run: streamlit run app.py")
