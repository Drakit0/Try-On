"""
Point Picker Tool for Garment Control Points
============================================

A Streamlit-based interactive tool to mark control points on garment images
and export them in the format required by garments.json.

Usage:
    streamlit run tools/point_picker.py

Features:
- Load PNG images from assets/garments/
- Click to add/edit control points
- Named points (L_shoulder_seam, R_shoulder_seam, etc.)
- Save/load points to/from JSON
- Export in control_points_src_px format
- Undo/redo functionality
- Visual preview with labels
"""

import streamlit as st
import json
from pathlib import Path
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
from typing import Dict, List, Tuple, Optional

# Constants
ASSETS_DIR = Path("assets/garments")
OUTPUT_DIR = Path("assets/garments/control_points")
DEFAULT_POINTS = [
    "L_shoulder_seam",
    "R_shoulder_seam", 
    "L_waist",
    "R_waist",
    "L_hip",
    "R_hip",
]

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_image_files() -> List[Path]:
    """Get list of PNG files in assets/garments directory."""
    if not ASSETS_DIR.exists():
        return []
    return sorted(ASSETS_DIR.glob("*.png"))


def load_image(image_path: Path) -> Image.Image:
    """Load and return PIL Image."""
    return Image.open(image_path)


def save_points_json(image_name: str, points: List[Dict], metadata: Optional[Dict] = None):
    """Save control points to JSON file."""
    output_file = OUTPUT_DIR / f"{image_name}_points.json"
    
    data = {
        "image": image_name,
        "tps_control_points": {
            "src": points,
            "description": "Control points for TPS warping: shoulders, waist, hips"
        }
    }
    
    if metadata:
        data["metadata"] = metadata
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_file


def load_points_json(image_name: str) -> Tuple[List[Dict], Optional[Dict]]:
    """Load control points from JSON file if exists."""
    input_file = OUTPUT_DIR / f"{image_name}_points.json"
    
    if input_file.exists():
        with open(input_file, 'r') as f:
            data = json.load(f)
            points = data.get("tps_control_points", {}).get("src", [])
            metadata = data.get("metadata", {})
            return points, metadata
    
    return [], {}


def export_to_garments_format(points: List[Dict], metadata: Dict) -> str:
    """Export points in the format used in garments.json."""
    output = {
        "metadata": metadata,
        "tps_control_points": {
            "src": points,
            "description": "Control points for TPS warping: shoulders, waist, hips"
        }
    }
    return json.dumps(output, indent=2)


def init_session_state():
    """Initialize Streamlit session state."""
    if 'points' not in st.session_state:
        st.session_state.points = []
    if 'current_point_index' not in st.session_state:
        st.session_state.current_point_index = 0
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'history_index' not in st.session_state:
        st.session_state.history_index = -1
    if 'metadata' not in st.session_state:
        st.session_state.metadata = {
            "base_shoulder_px": 280,
            "base_torso_px": 320,
            "y_offset_to_waist_px": 0
        }


def add_to_history():
    """Add current state to history for undo/redo."""
    # Remove any future history if we're in the middle
    st.session_state.history = st.session_state.history[:st.session_state.history_index + 1]
    
    # Add current state
    st.session_state.history.append(st.session_state.points.copy())
    st.session_state.history_index += 1
    
    # Limit history to 50 states
    if len(st.session_state.history) > 50:
        st.session_state.history.pop(0)
        st.session_state.history_index -= 1


def undo():
    """Undo last action."""
    if st.session_state.history_index > 0:
        st.session_state.history_index -= 1
        st.session_state.points = st.session_state.history[st.session_state.history_index].copy()


def redo():
    """Redo last undone action."""
    if st.session_state.history_index < len(st.session_state.history) - 1:
        st.session_state.history_index += 1
        st.session_state.points = st.session_state.history[st.session_state.history_index].copy()


def main():
    st.set_page_config(
        page_title="Garment Control Point Picker",
        page_icon="📍",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("📍 Garment Control Point Picker")
    st.markdown("*Interactive tool to mark control points on garment images for TPS warping*")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    
    # Image selection
    image_files = get_image_files()
    
    if not image_files:
        st.error(f"No PNG files found in `{ASSETS_DIR}`. Please add garment images first.")
        st.info("Expected location: `assets/garments/*.png`")
        return
    
    image_names = [f.stem for f in image_files]
    selected_name = st.sidebar.selectbox(
        "Select Garment Image",
        image_names,
        key="selected_image"
    )
    
    selected_file = ASSETS_DIR / f"{selected_name}.png"
    
    # Load/New buttons
    col1, col2 = st.sidebar.columns(2)
    
    if col1.button("📂 Load Points"):
        loaded_points, loaded_metadata = load_points_json(selected_name)
        if loaded_points:
            st.session_state.points = loaded_points
            st.session_state.metadata = loaded_metadata if loaded_metadata else st.session_state.metadata
            add_to_history()
            st.sidebar.success(f"Loaded {len(loaded_points)} points")
        else:
            st.sidebar.warning("No saved points found")
    
    if col2.button("🆕 New"):
        st.session_state.points = []
        st.session_state.current_point_index = 0
        st.session_state.history = []
        st.session_state.history_index = -1
        add_to_history()
        st.sidebar.info("Started new point set")
    
    # Metadata inputs
    st.sidebar.subheader("Metadata")
    st.session_state.metadata["base_shoulder_px"] = st.sidebar.number_input(
        "Base Shoulder (px)",
        min_value=0,
        value=st.session_state.metadata.get("base_shoulder_px", 280),
        step=10,
        help="Typical shoulder width in pixels"
    )
    st.session_state.metadata["base_torso_px"] = st.sidebar.number_input(
        "Base Torso (px)",
        min_value=0,
        value=st.session_state.metadata.get("base_torso_px", 320),
        step=10,
        help="Typical torso length in pixels"
    )
    st.session_state.metadata["y_offset_to_waist_px"] = st.sidebar.number_input(
        "Y Offset to Waist (px)",
        min_value=-100,
        max_value=100,
        value=st.session_state.metadata.get("y_offset_to_waist_px", 0),
        step=5,
        help="Vertical offset adjustment for waist alignment"
    )
    
    # Point name selection
    st.sidebar.subheader("Point Configuration")
    
    use_custom_points = st.sidebar.checkbox("Custom Point Names")
    
    if use_custom_points:
        num_points = st.sidebar.number_input("Number of Points", min_value=1, max_value=20, value=6)
        point_names = []
        for i in range(num_points):
            name = st.sidebar.text_input(f"Point {i+1} Name", value=f"point_{i+1}", key=f"point_name_{i}")
            point_names.append(name)
    else:
        point_names = DEFAULT_POINTS.copy()
        st.sidebar.info(f"Using default points: {', '.join(point_names)}")
    
    # Current point selection
    if len(st.session_state.points) < len(point_names):
        current_point_name = point_names[len(st.session_state.points)]
        st.sidebar.success(f"**Next point:** {current_point_name}")
    else:
        st.sidebar.success("✅ All points marked!")
    
    # Undo/Redo
    st.sidebar.subheader("History")
    col1, col2 = st.sidebar.columns(2)
    if col1.button("↶ Undo", disabled=st.session_state.history_index <= 0):
        undo()
    if col2.button("↷ Redo", disabled=st.session_state.history_index >= len(st.session_state.history) - 1):
        redo()
    
    # Main content area
    col_main, col_info = st.columns([2, 1])
    
    with col_main:
        st.subheader("Image Canvas")
        
        # Load image
        try:
            img = load_image(selected_file)
            img_width, img_height = img.size
            
            # Calculate display size (max 800px wide)
            max_width = 800
            if img_width > max_width:
                scale = max_width / img_width
                display_width = max_width
                display_height = int(img_height * scale)
            else:
                display_width = img_width
                display_height = img_height
                scale = 1.0
            
            st.caption(f"Image size: {img_width}x{img_height} px | Display scale: {scale:.2f}x")
            
            # Create canvas for point picking
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=img,
                update_streamlit=True,
                height=display_height,
                width=display_width,
                drawing_mode="point",
                point_display_radius=5,
                key="canvas",
            )
            
            # Process canvas clicks
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data.get("objects", [])
                
                # Check if new point was added
                if objects and len(objects) > len(st.session_state.points):
                    # Get the last point
                    new_point = objects[-1]
                    
                    if len(st.session_state.points) < len(point_names):
                        # Convert to original image coordinates
                        x = int(new_point["left"] / scale)
                        y = int(new_point["top"] / scale)
                        
                        point_data = {
                            "name": point_names[len(st.session_state.points)],
                            "x": x,
                            "y": y
                        }
                        
                        st.session_state.points.append(point_data)
                        add_to_history()
                        st.rerun()
            
            # Instructions
            st.info("👆 **Click on the image** to add control points in order. Points will be added according to the list on the right.")
            
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return
    
    with col_info:
        st.subheader("Control Points")
        
        # Display current points
        if st.session_state.points:
            st.markdown(f"**{len(st.session_state.points)}/{len(point_names)} points marked**")
            
            for i, point in enumerate(st.session_state.points):
                col1, col2 = st.columns([3, 1])
                col1.markdown(f"**{i+1}.** `{point['name']}`")
                col2.markdown(f"`({point['x']}, {point['y']})`")
                
                # Delete button
                if st.button("🗑️", key=f"delete_{i}"):
                    st.session_state.points.pop(i)
                    add_to_history()
                    st.rerun()
            
            st.markdown("---")
            
            # Edit point coordinates
            with st.expander("✏️ Edit Coordinates"):
                edit_idx = st.selectbox(
                    "Select point to edit",
                    range(len(st.session_state.points)),
                    format_func=lambda i: f"{st.session_state.points[i]['name']}"
                )
                
                if edit_idx is not None and edit_idx < len(st.session_state.points):
                    point = st.session_state.points[edit_idx]
                    
                    new_x = st.number_input("X coordinate", value=point['x'], step=1, key=f"edit_x_{edit_idx}")
                    new_y = st.number_input("Y coordinate", value=point['y'], step=1, key=f"edit_y_{edit_idx}")
                    
                    if st.button("Update"):
                        st.session_state.points[edit_idx]['x'] = new_x
                        st.session_state.points[edit_idx]['y'] = new_y
                        add_to_history()
                        st.success("Point updated!")
                        st.rerun()
        
        else:
            st.info("No points marked yet. Click on the image to start.")
        
        st.markdown("---")
        
        # Clear all button
        if st.button("🗑️ Clear All Points", type="secondary"):
            st.session_state.points = []
            add_to_history()
            st.rerun()
        
        # Save button
        if st.button("💾 Save Points", type="primary", disabled=len(st.session_state.points) == 0):
            output_file = save_points_json(selected_name, st.session_state.points, st.session_state.metadata)
            st.success(f"✅ Saved to:\n`{output_file}`")
        
        # Export section
        if st.session_state.points:
            st.markdown("---")
            st.subheader("📋 Export")
            
            export_format = export_to_garments_format(st.session_state.points, st.session_state.metadata)
            
            st.code(export_format, language="json")
            
            st.download_button(
                label="📥 Download JSON",
                data=export_format,
                file_name=f"{selected_name}_control_points.json",
                mime="application/json"
            )
            
            # Copy to clipboard helper
            st.caption("💡 Copy this JSON and paste into `assets/garments.json`")
    
    # Footer with stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Image", selected_name)
    col2.metric("Points Marked", f"{len(st.session_state.points)}/{len(point_names)}")
    col3.metric("Image Size", f"{img_width}x{img_height}")
    col4.metric("History", f"{st.session_state.history_index + 1}/{len(st.session_state.history)}")


if __name__ == "__main__":
    main()
