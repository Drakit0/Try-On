"""
Tests for Point Picker Tool
===========================

Test the point picker utility functions.
"""

import pytest
import json
from pathlib import Path
import sys
import tempfile
import shutil

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

# Mock streamlit_drawable_canvas if not available
try:
    import streamlit_drawable_canvas
except ImportError:
    sys.modules['streamlit_drawable_canvas'] = type(sys)('streamlit_drawable_canvas')
    sys.modules['streamlit_drawable_canvas'].st_canvas = lambda **kwargs: None

from point_picker import (
    save_points_json,
    load_points_json,
    export_to_garments_format,
)


@pytest.fixture
def temp_output_dir(monkeypatch):
    """Create temporary output directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    monkeypatch.setattr("point_picker.OUTPUT_DIR", temp_dir)
    
    yield temp_dir
    
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_points():
    """Sample control points for testing."""
    return [
        {"name": "L_shoulder_seam", "x": 60, "y": 50},
        {"name": "R_shoulder_seam", "x": 220, "y": 50},
        {"name": "L_waist", "x": 75, "y": 200},
        {"name": "R_waist", "x": 205, "y": 200},
        {"name": "L_hip", "x": 70, "y": 380},
        {"name": "R_hip", "x": 210, "y": 380}
    ]


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "base_shoulder_px": 240,
        "base_torso_px": 450,
        "y_offset_to_waist_px": -20
    }


class TestSaveLoadPoints:
    """Test saving and loading point data."""
    
    def test_save_points_creates_file(self, temp_output_dir, sample_points, sample_metadata):
        """Test that save_points_json creates a file."""
        image_name = "test_garment"
        output_file = save_points_json(image_name, sample_points, sample_metadata)
        
        assert output_file.exists()
        assert output_file.name == f"{image_name}_points.json"
    
    def test_save_points_correct_structure(self, temp_output_dir, sample_points, sample_metadata):
        """Test that saved JSON has correct structure."""
        image_name = "test_garment"
        output_file = save_points_json(image_name, sample_points, sample_metadata)
        
        with open(output_file) as f:
            data = json.load(f)
        
        assert "image" in data
        assert data["image"] == image_name
        assert "tps_control_points" in data
        assert "src" in data["tps_control_points"]
        assert "description" in data["tps_control_points"]
        assert "metadata" in data
    
    def test_save_points_preserves_data(self, temp_output_dir, sample_points, sample_metadata):
        """Test that saved data matches input."""
        image_name = "test_garment"
        output_file = save_points_json(image_name, sample_points, sample_metadata)
        
        with open(output_file) as f:
            data = json.load(f)
        
        assert data["tps_control_points"]["src"] == sample_points
        assert data["metadata"] == sample_metadata
    
    def test_load_points_nonexistent_file(self, temp_output_dir):
        """Test loading from nonexistent file returns empty."""
        points, metadata = load_points_json("nonexistent")
        
        assert points == []
        assert metadata == {}
    
    def test_load_points_existing_file(self, temp_output_dir, sample_points, sample_metadata):
        """Test loading from existing file."""
        image_name = "test_garment"
        save_points_json(image_name, sample_points, sample_metadata)
        
        loaded_points, loaded_metadata = load_points_json(image_name)
        
        assert loaded_points == sample_points
        assert loaded_metadata == sample_metadata
    
    def test_save_load_roundtrip(self, temp_output_dir, sample_points, sample_metadata):
        """Test that save->load preserves data."""
        image_name = "test_garment"
        
        save_points_json(image_name, sample_points, sample_metadata)
        loaded_points, loaded_metadata = load_points_json(image_name)
        
        assert loaded_points == sample_points
        assert loaded_metadata == sample_metadata


class TestExportFormat:
    """Test export formatting."""
    
    def test_export_format_structure(self, sample_points, sample_metadata):
        """Test export has correct structure."""
        export_str = export_to_garments_format(sample_points, sample_metadata)
        data = json.loads(export_str)
        
        assert "metadata" in data
        assert "tps_control_points" in data
        assert "src" in data["tps_control_points"]
        assert "description" in data["tps_control_points"]
    
    def test_export_format_valid_json(self, sample_points, sample_metadata):
        """Test export is valid JSON."""
        export_str = export_to_garments_format(sample_points, sample_metadata)
        
        # Should not raise exception
        json.loads(export_str)
    
    def test_export_format_preserves_data(self, sample_points, sample_metadata):
        """Test export preserves all data."""
        export_str = export_to_garments_format(sample_points, sample_metadata)
        data = json.loads(export_str)
        
        assert data["metadata"] == sample_metadata
        assert data["tps_control_points"]["src"] == sample_points
    
    def test_export_format_formatted(self, sample_points, sample_metadata):
        """Test export is properly formatted (indented)."""
        export_str = export_to_garments_format(sample_points, sample_metadata)
        
        # Should have newlines and indentation
        assert "\n" in export_str
        assert "  " in export_str


class TestPointValidation:
    """Test point data validation."""
    
    def test_point_has_required_fields(self, sample_points):
        """Test that points have required fields."""
        for point in sample_points:
            assert "name" in point
            assert "x" in point
            assert "y" in point
    
    def test_point_coordinates_are_numbers(self, sample_points):
        """Test that coordinates are numeric."""
        for point in sample_points:
            assert isinstance(point["x"], (int, float))
            assert isinstance(point["y"], (int, float))
    
    def test_point_names_are_strings(self, sample_points):
        """Test that point names are strings."""
        for point in sample_points:
            assert isinstance(point["name"], str)
            assert len(point["name"]) > 0


class TestMetadataValidation:
    """Test metadata validation."""
    
    def test_metadata_has_required_fields(self, sample_metadata):
        """Test metadata has required fields."""
        assert "base_shoulder_px" in sample_metadata
        assert "base_torso_px" in sample_metadata
        assert "y_offset_to_waist_px" in sample_metadata
    
    def test_metadata_values_are_numbers(self, sample_metadata):
        """Test metadata values are numeric."""
        assert isinstance(sample_metadata["base_shoulder_px"], (int, float))
        assert isinstance(sample_metadata["base_torso_px"], (int, float))
        assert isinstance(sample_metadata["y_offset_to_waist_px"], (int, float))


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_points_list(self, temp_output_dir, sample_metadata):
        """Test saving empty points list."""
        output_file = save_points_json("test", [], sample_metadata)
        
        with open(output_file) as f:
            data = json.load(f)
        
        assert data["tps_control_points"]["src"] == []
    
    def test_single_point(self, temp_output_dir, sample_metadata):
        """Test saving single point."""
        points = [{"name": "test_point", "x": 100, "y": 200}]
        output_file = save_points_json("test", points, sample_metadata)
        
        loaded_points, _ = load_points_json("test")
        assert len(loaded_points) == 1
        assert loaded_points[0]["name"] == "test_point"
    
    def test_many_points(self, temp_output_dir, sample_metadata):
        """Test saving many points."""
        points = [
            {"name": f"point_{i}", "x": i * 10, "y": i * 20}
            for i in range(20)
        ]
        
        save_points_json("test", points, sample_metadata)
        loaded_points, _ = load_points_json("test")
        
        assert len(loaded_points) == 20
    
    def test_unicode_in_names(self, temp_output_dir, sample_metadata):
        """Test unicode characters in point names."""
        points = [{"name": "左肩", "x": 100, "y": 200}]
        
        save_points_json("test", points, sample_metadata)
        loaded_points, _ = load_points_json("test")
        
        assert loaded_points[0]["name"] == "左肩"
    
    def test_negative_coordinates(self, temp_output_dir, sample_metadata):
        """Test negative coordinates (edge case)."""
        points = [{"name": "test", "x": -10, "y": -20}]
        
        save_points_json("test", points, sample_metadata)
        loaded_points, _ = load_points_json("test")
        
        assert loaded_points[0]["x"] == -10
        assert loaded_points[0]["y"] == -20


def test_output_directory_creation(temp_output_dir):
    """Test that output directory is created if it doesn't exist."""
    # Remove the directory
    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir)
    
    # Create it again (simulating first run)
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    
    assert temp_output_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
