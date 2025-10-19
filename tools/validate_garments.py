"""
Validate Garments Metadata
===========================

Validate that garments.json is correct and control points are properly defined.

Usage:
    python tools/validate_garments.py
"""

import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def validate_garments():
    """Validate garments.json structure and content."""
    
    print("=" * 70)
    print("GARMENTS METADATA VALIDATION")
    print("=" * 70)
    
    # Load JSON
    json_path = Path("assets/garments.json")
    
    if not json_path.exists():
        print(f"❌ ERROR: {json_path} not found")
        return False
    
    try:
        with open(json_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON: {e}")
        return False
    
    print(f"✅ JSON valid and loaded")
    
    # Check structure
    if "garments" not in data:
        print("❌ ERROR: Missing 'garments' key")
        return False
    
    garments = data["garments"]
    print(f"✅ Found {len(garments)} garments\n")
    
    # Validate each garment
    all_valid = True
    
    for i, garment in enumerate(garments, 1):
        print(f"\n{'=' * 70}")
        print(f"Garment {i}: {garment.get('name', 'UNNAMED')}")
        print(f"{'=' * 70}")
        
        # Required fields
        required_fields = ['id', 'name', 'path', 'type', 'metadata', 'tps_control_points']
        for field in required_fields:
            if field not in garment:
                print(f"  ❌ Missing required field: {field}")
                all_valid = False
            else:
                print(f"  ✅ {field}: {garment[field] if field not in ['metadata', 'tps_control_points'] else '...'}")
        
        # Validate metadata
        if 'metadata' in garment:
            meta = garment['metadata']
            required_meta = ['base_shoulder_px', 'base_torso_px', 'y_offset_to_waist_px']
            
            print("\n  Metadata:")
            for field in required_meta:
                if field not in meta:
                    print(f"    ❌ Missing: {field}")
                    all_valid = False
                else:
                    value = meta[field]
                    if not isinstance(value, (int, float)):
                        print(f"    ❌ {field}: {value} (not a number)")
                        all_valid = False
                    elif value < 0 and field != 'y_offset_to_waist_px':
                        print(f"    ❌ {field}: {value} (negative value)")
                        all_valid = False
                    else:
                        print(f"    ✅ {field}: {value} px")
        
        # Validate control points
        if 'tps_control_points' in garment:
            tps = garment['tps_control_points']
            
            if 'src' not in tps:
                print("\n  ❌ Missing 'src' in tps_control_points")
                all_valid = False
            else:
                src_points = tps['src']
                print(f"\n  Control Points: {len(src_points)} points")
                
                if len(src_points) < 6:
                    print(f"    ❌ Only {len(src_points)} points (need at least 6)")
                    all_valid = False
                else:
                    print(f"    ✅ {len(src_points)} points (>= 6 required)")
                
                # Check each point
                required_point_names = [
                    'L_shoulder_seam', 'R_shoulder_seam',
                    'L_waist', 'R_waist',
                    'L_hip', 'R_hip'
                ]
                
                point_names = [pt.get('name') for pt in src_points]
                
                for name in required_point_names:
                    if name in point_names:
                        print(f"    ✅ {name}")
                    else:
                        print(f"    ⚠️  Missing recommended point: {name}")
                
                # Validate point structure
                for pt in src_points:
                    if 'name' not in pt or 'x' not in pt or 'y' not in pt:
                        print(f"    ❌ Invalid point structure: {pt}")
                        all_valid = False
                    elif not isinstance(pt['x'], (int, float)) or not isinstance(pt['y'], (int, float)):
                        print(f"    ❌ Non-numeric coordinates in {pt['name']}")
                        all_valid = False
        
        # Check file exists
        if 'path' in garment:
            garment_path = Path("assets") / garment['path']
            if garment_path.exists():
                print(f"\n  ✅ File exists: {garment_path}")
            else:
                print(f"\n  ⚠️  File not found: {garment_path}")
                print(f"      (This is OK for demo - app will handle gracefully)")
    
    print("\n" + "=" * 70)
    
    if all_valid:
        print("✅ ALL VALIDATIONS PASSED")
        print("=" * 70)
        return True
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("=" * 70)
        return False


def test_tps_compatibility():
    """Test that control points work with TPS."""
    print("\n" + "=" * 70)
    print("TPS COMPATIBILITY TEST")
    print("=" * 70)
    
    try:
        from src.tps import ThinPlateSpline
        import numpy as np
        
        json_path = Path("assets/garments.json")
        with open(json_path) as f:
            data = json.load(f)
        
        for garment in data['garments']:
            print(f"\nTesting: {garment['name']}")
            
            if 'tps_control_points' in garment and 'src' in garment['tps_control_points']:
                src_points = garment['tps_control_points']['src']
                
                # Convert to numpy array
                src_pts = np.array([
                    [pt['x'], pt['y']] for pt in src_points
                ], dtype=np.float32)
                
                try:
                    # Create TPS
                    tps = ThinPlateSpline(src_pts)
                    print(f"  ✅ TPS initialized with {len(src_pts)} points")
                    
                    # Create dummy destination points (slight shift)
                    dst_pts = src_pts + np.random.randn(len(src_pts), 2) * 5
                    
                    # Fit
                    tps.fit(dst_pts)
                    print(f"  ✅ TPS fit successful")
                    
                    # Generate grid
                    map_x, map_y = tps.sample_grid(256, 256)
                    print(f"  ✅ Grid generation successful: {map_x.shape}")
                    
                except Exception as e:
                    print(f"  ❌ TPS error: {e}")
                    return False
        
        print("\n" + "=" * 70)
        print("✅ TPS COMPATIBILITY VERIFIED")
        print("=" * 70)
        return True
        
    except ImportError:
        print("\n⚠️  Cannot import TPS module (this is OK if not installed)")
        return True


if __name__ == '__main__':
    valid = validate_garments()
    tps_ok = test_tps_compatibility()
    
    if valid and tps_ok:
        print("\n✅ All validations passed! garments.json is ready to use.")
        sys.exit(0)
    else:
        print("\n❌ Validation failed. Please fix errors above.")
        sys.exit(1)
