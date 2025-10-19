"""
Control Points Validator
========================

Validates control points created with the point picker tool.

Usage:
    python tools/validate_control_points.py
    
Checks:
- Point count (minimum 6)
- Required point names
- Coordinate ranges (within image bounds)
- Symmetry (L/R pairs)
- Metadata values
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image


CONTROL_POINTS_DIR = Path("assets/garments/control_points")
GARMENTS_DIR = Path("assets/garments")
REQUIRED_POINTS = ["L_shoulder_seam", "R_shoulder_seam", "L_waist", "R_waist", "L_hip", "R_hip"]


def load_control_points(json_file: Path) -> Tuple[Dict, bool]:
    """Load and validate control points JSON."""
    try:
        with open(json_file) as f:
            data = json.load(f)
        return data, True
    except json.JSONDecodeError as e:
        print(f"  ❌ Invalid JSON: {e}")
        return {}, False
    except Exception as e:
        print(f"  ❌ Error loading file: {e}")
        return {}, False


def validate_structure(data: Dict) -> bool:
    """Validate JSON structure."""
    errors = []
    
    if "tps_control_points" not in data:
        errors.append("Missing 'tps_control_points' key")
    else:
        tps = data["tps_control_points"]
        if "src" not in tps:
            errors.append("Missing 'src' key in tps_control_points")
        if "description" not in tps:
            errors.append("Missing 'description' key in tps_control_points")
    
    if "metadata" not in data:
        errors.append("Missing 'metadata' key")
    else:
        metadata = data["metadata"]
        required_meta = ["base_shoulder_px", "base_torso_px", "y_offset_to_waist_px"]
        for key in required_meta:
            if key not in metadata:
                errors.append(f"Missing metadata key: {key}")
    
    if errors:
        for error in errors:
            print(f"  ❌ {error}")
        return False
    
    return True


def validate_points(points: List[Dict], image_name: str) -> Tuple[bool, List[str]]:
    """Validate control points."""
    errors = []
    warnings = []
    
    # Check point count
    if len(points) < 6:
        errors.append(f"Too few points: {len(points)} (minimum 6)")
    
    # Check required point names
    point_names = [p["name"] for p in points]
    missing = set(REQUIRED_POINTS) - set(point_names)
    if missing:
        errors.append(f"Missing required points: {missing}")
    
    # Check point structure
    for i, point in enumerate(points):
        if "name" not in point:
            errors.append(f"Point {i}: missing 'name'")
        if "x" not in point:
            errors.append(f"Point {i}: missing 'x' coordinate")
        if "y" not in point:
            errors.append(f"Point {i}: missing 'y' coordinate")
        
        # Check coordinate types
        if "x" in point and not isinstance(point["x"], (int, float)):
            errors.append(f"Point {i} ({point.get('name', '?')}): x is not numeric")
        if "y" in point and not isinstance(point["y"], (int, float)):
            errors.append(f"Point {i} ({point.get('name', '?')}): y is not numeric")
    
    # Check coordinates within image bounds
    image_path = GARMENTS_DIR / f"{image_name}.png"
    if image_path.exists():
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        for point in points:
            if "x" in point and "y" in point:
                x, y = point["x"], point["y"]
                name = point.get("name", "?")
                
                if x < 0 or x >= img_width:
                    errors.append(f"Point '{name}': x={x} outside image width (0-{img_width})")
                if y < 0 or y >= img_height:
                    errors.append(f"Point '{name}': y={y} outside image height (0-{img_height})")
    else:
        warnings.append(f"Image not found: {image_path} (cannot validate bounds)")
    
    # Check symmetry (L/R pairs)
    for base_name in ["shoulder_seam", "waist", "hip"]:
        l_name = f"L_{base_name}"
        r_name = f"R_{base_name}"
        
        l_points = [p for p in points if p["name"] == l_name]
        r_points = [p for p in points if p["name"] == r_name]
        
        if l_points and r_points:
            l_point = l_points[0]
            r_point = r_points[0]
            
            # Check Y coordinates are similar (should be on same horizontal line)
            y_diff = abs(l_point["y"] - r_point["y"])
            if y_diff > 20:
                warnings.append(
                    f"{l_name}/{r_name}: Y coordinates differ by {y_diff}px "
                    f"(L={l_point['y']}, R={r_point['y']}). Should be similar."
                )
            
            # Check X order (L should be left of R)
            if l_point["x"] >= r_point["x"]:
                warnings.append(
                    f"{l_name}/{r_name}: L point (x={l_point['x']}) should be left of R point (x={r_point['x']})"
                )
    
    return len(errors) == 0, errors + warnings


def validate_metadata(metadata: Dict) -> Tuple[bool, List[str]]:
    """Validate metadata values."""
    errors = []
    warnings = []
    
    # Check base_shoulder_px
    shoulder = metadata.get("base_shoulder_px", 0)
    if shoulder <= 0:
        errors.append(f"base_shoulder_px must be positive (got {shoulder})")
    elif shoulder < 150 or shoulder > 500:
        warnings.append(f"base_shoulder_px={shoulder} is unusual (typical: 220-300)")
    
    # Check base_torso_px
    torso = metadata.get("base_torso_px", 0)
    if torso <= 0:
        errors.append(f"base_torso_px must be positive (got {torso})")
    elif torso < 200 or torso > 800:
        warnings.append(f"base_torso_px={torso} is unusual (typical: 300-500)")
    
    # Check y_offset_to_waist_px
    y_offset = metadata.get("y_offset_to_waist_px", 0)
    if abs(y_offset) > 100:
        warnings.append(f"y_offset_to_waist_px={y_offset} is large (typical: -50 to +50)")
    
    return len(errors) == 0, errors + warnings


def validate_file(json_file: Path) -> bool:
    """Validate a single control points file."""
    print(f"\n📄 {json_file.name}")
    
    # Load JSON
    data, loaded = load_control_points(json_file)
    if not loaded:
        return False
    
    # Get image name
    image_name = json_file.stem.replace("_points", "")
    
    # Validate structure
    if not validate_structure(data):
        return False
    
    # Validate points
    points = data.get("tps_control_points", {}).get("src", [])
    points_valid, point_issues = validate_points(points, image_name)
    
    # Validate metadata
    metadata = data.get("metadata", {})
    metadata_valid, meta_issues = validate_metadata(metadata)
    
    # Print issues
    all_valid = points_valid and metadata_valid
    
    if point_issues:
        for issue in point_issues:
            if "❌" not in issue and "⚠️" not in issue:
                if any(word in issue.lower() for word in ["missing", "outside", "too few", "not numeric"]):
                    print(f"  ❌ {issue}")
                else:
                    print(f"  ⚠️  {issue}")
    
    if meta_issues:
        for issue in meta_issues:
            if "must be" in issue.lower():
                print(f"  ❌ {issue}")
            else:
                print(f"  ⚠️  {issue}")
    
    # Summary
    if all_valid and not point_issues and not meta_issues:
        print(f"  ✅ Valid ({len(points)} points)")
    elif all_valid:
        print(f"  ✅ Valid with warnings ({len(points)} points)")
    else:
        print(f"  ❌ Invalid")
    
    return all_valid


def main():
    """Main validation routine."""
    print("=" * 60)
    print("Control Points Validator")
    print("=" * 60)
    
    # Check if directory exists
    if not CONTROL_POINTS_DIR.exists():
        print(f"\n❌ Directory not found: {CONTROL_POINTS_DIR}")
        print("   Run the point picker tool first to create control points.")
        return
    
    # Get all JSON files
    json_files = sorted(CONTROL_POINTS_DIR.glob("*_points.json"))
    
    if not json_files:
        print(f"\n⚠️  No control point files found in {CONTROL_POINTS_DIR}")
        print("   Use the point picker tool to create some.")
        return
    
    print(f"\nFound {len(json_files)} control point file(s)")
    
    # Validate each file
    valid_count = 0
    for json_file in json_files:
        if validate_file(json_file):
            valid_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Summary: {valid_count}/{len(json_files)} files valid")
    print("=" * 60)
    
    if valid_count == len(json_files):
        print("\n✅ All control point files are valid!")
    else:
        print(f"\n⚠️  {len(json_files) - valid_count} file(s) have errors or warnings")
    
    # Check for corresponding images
    print("\n" + "=" * 60)
    print("Image Verification")
    print("=" * 60)
    
    for json_file in json_files:
        image_name = json_file.stem.replace("_points", "")
        image_path = GARMENTS_DIR / f"{image_name}.png"
        
        if image_path.exists():
            print(f"✅ {image_name}.png - Found")
        else:
            print(f"❌ {image_name}.png - Not found in {GARMENTS_DIR}")


if __name__ == "__main__":
    main()
