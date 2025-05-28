import matplotlib
matplotlib.use('Agg') # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt

import os
from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename
import svgpathtools
from shapely.geometry import LineString, Polygon, MultiPolygon, Point
from shapely import affinity
from shapely.ops import unary_union, nearest_points
import trimesh
from trimesh.creation import extrude_polygon
import re
import networkx as nx
import numpy as np
import math
from io import BytesIO
import base64

# Import vedo and Pillow for STL preview generation
import vedo
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
ALLOWED_EXTENSIONS = {'svg'}
app.secret_key = 'supersecretkey' # Used for flashing messages

REFERENCE_TARGET_X_SIZE = 80.0
DEFAULT_REFERENCE_CUTOUT_AREA_THRESHOLD = 10.0

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """
    Checks if the uploaded file has an allowed SVG extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === Hilfsfunktionen ===
def extrude_any_shape(geometry, height):
    """
    Extrudes a Shapely Polygon or MultiPolygon into a 3D trimesh.
    Handles both single and multiple polygons by concatenating their meshes.
    """
    meshes = []
    if isinstance(geometry, Polygon):
        # Extrude a single polygon
        meshes.append(extrude_polygon(geometry, height=height))
    elif isinstance(geometry, MultiPolygon):
        # Iterate through multiple polygons in a MultiPolygon and extrude each
        for poly in geometry.geoms:
            meshes.append(extrude_polygon(poly, height=height))
    else:
        # Raise an error for unsupported geometry types
        raise ValueError("Nicht unterstützter Geometrietyp für die Extrusion.")
    return trimesh.util.concatenate(meshes)

def sample_svg_path(path_segment, resolution=0.5):
    """
    Samples points along an SVG path segment to convert it into a series of discrete coordinates.
    This is essential for converting vector graphics into a format suitable for polygon creation.
    `resolution` controls the density of sampled points (smaller values mean smoother curves).
    """
    length = path_segment.length()
    # Ensure at least 2 samples for a segment
    num_samples = max(int(length / resolution), 2)
    
    # Generate points along the path segment
    return [path_segment.point(t) for t in [i / (num_samples - 1) for i in range(num_samples)]]

def parse_svg_length(length_str, dpi=96):
    """
    Parses an SVG length string (e.g., "100px", "20mm", "1in") and converts it to pixels.
    Supports various SVG units.
    """
    # Define conversion units relative to pixels (at 96 DPI)
    units = {
        'px': 1,
        'mm': dpi / 25.4,
        'cm': dpi / 2.54,
        'in': dpi,
        'pt': dpi / 72,
        'pc': dpi / 6,
        '': 1 # Default unit if none specified (usually pixels)
    }
    
    # Regex to extract numerical value and unit
    match = re.fullmatch(r"([0-9.+-eE]+)([a-z%]*)", length_str.strip())
    if not match:
        try:
            # If no unit, try to parse as a float directly
            return float(length_str)
        except ValueError:
            raise ValueError(f"SVG-Länge konnte nicht analysiert werden: {length_str}")
    
    value, unit = match.groups()
    # Convert value based on the detected unit
    return float(value) * units.get(unit.lower(), 1)

def parse_transform(transform_string):
    """
    Parses an SVG 'transform' attribute string into a list of individual transformations.
    Supports 'translate', 'scale', 'rotate', and 'matrix' transformations.
    """
    transforms = []
    if not transform_string:
        return transforms
    
    # Regex to find individual transform functions and their values
    transform_regex = re.compile(r'(translate|scale|rotate|matrix)\(([^)]*)\)')
    for match in transform_regex.finditer(transform_string):
        transform_type = match.group(1)
        values_str = match.group(2)
        values = [float(v) for v in values_str.split(',')] # Convert values to floats

        if transform_type == "translate":
            # Translate: (tx, ty) - ty defaults to 0 if not provided
            transforms.append(("translate", values[0], values[1] if len(values) > 1 else 0))
        elif transform_type == "scale":
            # Scale: (sx, sy) - sy defaults to sx if not provided
            transforms.append(("scale", values[0], values[1] if len(values) > 1 else values[0]))
        elif transform_type == "rotate":
            # Rotate: (angle) or (angle, cx, cy)
            if len(values) == 1:
                transforms.append(("rotate", values[0]))
            elif len(values) == 3:
                transforms.append(("rotate", values[0], values[1], values[2]))
            else:
                print(f"Warnung: Unerwartete Anzahl von Werten für die Rotations-Transformation: {values}")
        elif transform_type == "matrix":
            # Matrix: (a, b, c, d, e, f)
            if len(values) == 6:
                transforms.append(("matrix", values))
            else:
                print(f"Warnung: Unerwartete Anzahl von Werten für die Matrix-Transformation: {values}")
    return transforms

def apply_transform(x, y, transform_type, values):
    """
    Applies a single SVG transformation (translate, scale, rotate, or matrix)
    to a given (x, y) coordinate pair.
    """
    if transform_type == "translate":
        tx, ty = values[0], values[1]
        return x + tx, y + ty
    elif transform_type == "scale":
        sx, sy = values[0], values[1]
        return x * sx, y * sy
    elif transform_type == "rotate":
        angle_rad = math.radians(values[0])
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)
        if len(values) == 3: # Rotate around a specific point (cx, cy)
            cx, cy = values[1], values[2]
            translated_x = x - cx
            translated_y = y - cy
            rotated_x = translated_x * cos_theta - translated_y * sin_theta
            rotated_y = translated_x * sin_theta + translated_y * cos_theta
            return rotated_x + cx, rotated_y + cy
        else: # Rotate around origin (0,0)
            return x * cos_theta - y * sin_theta, x * sin_theta + y * cos_theta
    elif transform_type == "matrix":
        a, b, c, d, e, f = values
        return a * x + c * y + e, b * x + d * y + f
    return x, y # Return original if transform type is not recognized

def generate_bridge_connections(polygons, extra_bridge_factor=0.0):
    """
    Generates structural bridges between a list of Shapely polygons using a Minimum Spanning Tree (MST)
    approach to ensure connectivity, and then adds additional bridges based on a factor.
    """
    if len(polygons) <= 1:
        return [] # No bridges needed for 0 or 1 polygon

    all_potential_bridges = []
    # Store polygon index for MST algorithm
    polygon_indices = {poly: i for i, poly in enumerate(polygons)}

    # Calculate all potential bridges between polygon pairs
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            poly1 = polygons[i]
            poly2 = polygons[j]
            # Find the nearest points between the two polygons
            p1_nearest, p2_nearest = nearest_points(poly1, poly2)
            dist = p1_nearest.distance(p2_nearest)
            all_potential_bridges.append({'u': i, 'v': j, 'weight': dist, 'p1': p1_nearest, 'p2': p2_nearest})
    
    # Sort potential bridges by distance (weight)
    all_potential_bridges.sort(key=lambda x: x['weight'])

    mst_bridges = []
    mst_edges_set = set() # To keep track of edges already added by MST
    parent = list(range(len(polygons))) # For Union-Find data structure

    # Union-Find helper functions
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True # Union happened
        return False # Already in the same set

    num_components = len(polygons)
    # Kruskal's algorithm for MST
    for bridge_data in all_potential_bridges:
        u, v, weight, p1, p2 = bridge_data['u'], bridge_data['v'], bridge_data['weight'], bridge_data['p1'], bridge_data['p2']
        if union(u, v):
            mst_bridges.append(LineString([p1, p2]))
            mst_edges_set.add(tuple(sorted((u, v)))) # Add sorted tuple to set for uniqueness
            num_components -= 1
            if num_components == 1: # All components are connected
                break
    
    # Add additional bridges based on the extra_bridge_factor
    additional_bridges = []
    num_additional_to_add = int(len(all_potential_bridges) * extra_bridge_factor)
    added_count = 0
    for bridge_data in all_potential_bridges:
        u, v, weight, p1, p2 = bridge_data['u'], bridge_data['v'], bridge_data['weight'], bridge_data['p1'], bridge_data['p2']
        if tuple(sorted((u, v))) not in mst_edges_set: # Only add if not already part of MST
            additional_bridges.append(LineString([p1, p2]))
            added_count += 1
            if added_count >= num_additional_to_add:
                break
    
    return mst_bridges + additional_bridges

def ensure_minimum_bridges(outer_poly, inner_polys, num_points_on_inner_perimeter=4):
    """
    Ensures a minimum number of bridges connect each inner polygon to the main outer polygon.
    This is important for stabilizing inner details of the cookie cutter.
    """
    bridges = []
    used_pairs = set() # To avoid duplicate bridges
    
    # Filter out the outer polygon itself if it's mistakenly in inner_polys
    actual_inner_polys = [p for p in inner_polys if not p.equals(outer_poly)]

    for inner in actual_inner_polys:
        perimeter_length = inner.exterior.length
        if perimeter_length == 0:
            continue # Skip empty polygons

        # Calculate step length to distribute points evenly on the perimeter
        step_length = perimeter_length / num_points_on_inner_perimeter
        current_distance = 0.0

        for i in range(num_points_on_inner_perimeter):
            # Interpolate a point on the inner polygon's exterior
            p_inner = inner.exterior.interpolate(current_distance)
            # Find the nearest point on the outer polygon to this inner point
            p_outer, _ = nearest_points(outer_poly, p_inner)
            
            bridge = LineString([p_inner, p_outer])
            
            # Add bridge if it's valid and not a zero-length line
            if not bridge.is_empty and bridge.length > 1e-6:
                # Create a canonical key for the bridge to check for duplicates
                coords = sorted([(round(p_inner.x, 3), round(p_inner.y, 3)), (round(p_outer.x, 3), round(p_outer.y, 3))])
                key = tuple(coords[0] + coords[1])
                if key not in used_pairs:
                    bridges.append(bridge)
                    used_pairs.add(key)
            current_distance += step_length
    return bridges

def fill_small_cutouts(polygon, area_threshold=5.0):
    """
    Identifies and "fills" (removes) small interior holes (cutouts) within a polygon
    if their area is below a specified threshold.
    Returns the cleaned polygon and a list of the geometries that were removed.
    """
    removed_hole_geometries = []

    def filter_holes(poly):
        nonlocal removed_hole_geometries
        valid_interiors = []
        for ring in poly.interiors:
            hole_poly = Polygon(ring)
            if hole_poly.area < area_threshold:
                removed_hole_geometries.append(hole_poly) # Collect removed holes
            else:
                valid_interiors.append(ring) # Keep larger holes
        return Polygon(poly.exterior, valid_interiors)

    if isinstance(polygon, Polygon):
        cleaned_poly = filter_holes(polygon)
        return cleaned_poly, removed_hole_geometries
    elif isinstance(polygon, MultiPolygon):
        new_polys = []
        removed_holes_multi = []
        for poly in polygon.geoms:
            cleaned, removed = filter_holes(poly)
            new_polys.append(cleaned)
            removed_holes_multi.extend(removed)
        return MultiPolygon(new_polys), removed_holes_multi
    else:
        # If the input is not a Polygon or MultiPolygon, return as is with no removed holes
        return polygon, removed_hole_geometries

def list_internal_cutouts(normalized_outlines):
    """
    Analyzes a list of normalized (translated to origin) Shapely LineStrings
    to identify potential internal cutouts and their areas.
    This helps in determining which parts might be holes in the final cutter.
    """
    polygons = []
    # Convert LineStrings to Polygons for area calculation and containment checks
    for outline in normalized_outlines:
        try:
            poly = Polygon(outline)
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)
        except Exception as e:
            print(f"Warnung: Polygon konnte aus Umriss für Ausschnittserkennung nicht gebildet werden (übersprungen): {e}")
            continue

    if not polygons:
        return [] # No polygons to check

    valid_polygons = [p for p in polygons if isinstance(p, Polygon)] # Corrected syntax here
    if not valid_polygons:
        return []

    # The largest polygon is typically the main outer boundary
    main_outer_poly = max(valid_polygons, key=lambda p: p.area)
    
    cutout_areas = []
    for poly in valid_polygons:
        if poly.equals(main_outer_poly):
            # If it's the main outer polygon, its interiors are holes
            for interior_ring in poly.interiors:
                hole_poly = Polygon(interior_ring)
                cutout_areas.append(hole_poly.area)
        else:
            # If it's another polygon, check if it's contained within the main outer
            if main_outer_poly.contains(poly):
                cutout_areas.append(poly.area)
                # Also check for holes within these inner polygons
                for interior_ring in poly.interiors:
                    hole_poly = Polygon(interior_ring)
                    cutout_areas.append(hole_poly.area)
    return cutout_areas

def plot_cutter_preview(normalized_outlines, removed_hole_geometries, wall_thickness, current_cutout_threshold, target_x, output_base_name, outer_base_shape_buffered, output_path):
    """
    Generates a 2D preview of the cookie cutter, showing walls and filled/unfilled cutouts.
    - Shows the outer base (light gray).
    - Shows the cutter walls (black lines).
    - Highlights filled cutouts in red.
    - Highlights *unfilled* cutouts in green.
    - Displays cutout areas on the image.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#e0e0e0') # Set a light gray background for the figure

    # Draw the outer base if it exists
    if outer_base_shape_buffered:
        x, y = outer_base_shape_buffered.exterior.xy
        ax.fill(x, y, color='lightgray', alpha=0.5, label='Äußere Basis')
        for interior in outer_base_shape_buffered.interiors:
            x, y = interior.xy
            ax.fill(x, y, color='white', alpha=0.5)

    # Identify the main outer shape from normalized_outlines for containment checks
    main_outer_outline_poly = None
    max_area = -1
    for outline in normalized_outlines:
        try:
            poly = Polygon(outline)
            if poly.is_valid and poly.area > max_area:
                max_area = poly.area
                main_outer_outline_poly = poly
        except Exception:
            pass # Skip invalid polygons

    # Plot the main outer cutter wall (exterior only)
    if main_outer_outline_poly:
        # Create a buffered wall shape for the main outer outline for drawing its exterior
        main_outer_wall_shape = main_outer_outline_poly.buffer(wall_thickness / 2.0, cap_style=1, join_style=1, resolution=16)
        if main_outer_wall_shape.is_valid and not main_outer_wall_shape.is_empty:
            x, y = main_outer_wall_shape.exterior.xy
            ax.plot(x, y, color='black', linewidth=1.5, label='Ausstecherwand')
    
    # Iterate through all normalized_outlines to plot cutouts (filled or unfilled)
    if main_outer_outline_poly:
        for outline in normalized_outlines:
            try:
                cutout_poly = Polygon(outline) # This is the original SVG shape polygon
                if not cutout_poly.is_valid or cutout_poly.is_empty:
                    continue

                # Case 1: This outline is an internal cutout (e.g., a letter, or a hole within a letter)
                if main_outer_outline_poly.contains(cutout_poly) and not main_outer_outline_poly.equals(cutout_poly):
                    
                    # The decision to fill is based on the area of the *original cutout polygon*
                    # compared to the threshold. This aligns with how fill_small_cutouts works.
                    is_filled_cutout_by_threshold = (cutout_poly.area < current_cutout_threshold)
                    
                    x, y = cutout_poly.exterior.xy
                    area_text = f"{cutout_poly.area:.1f}"

                    if is_filled_cutout_by_threshold:
                        # Draw filled cutouts in red
                        ax.fill(x, y, color='red', alpha=0.7, label='Gefüllter Ausschnitt' if 'Gefüllter Ausschnitt' not in [l.get_label() for l in ax.get_lines()] else "")
                        text_color = 'black'
                    else:
                        # Draw unfilled cutouts in green dashed outline
                        ax.plot(x, y, color='green', linestyle='--', linewidth=0.8,
                                label='Ungefüllter Ausschnitt' if 'Ungefüllter Ausschnitt' not in [l.get_label() for l in ax.get_lines()] else "")
                        text_color = 'green'
                    
                    if cutout_poly.centroid.is_valid:
                        ax.text(cutout_poly.centroid.x, cutout_poly.centroid.y, area_text,
                                fontsize=10, ha='center', va='center', color=text_color, weight='bold',
                                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

                # Case 2: This outline IS the main outer shape, and we need to check its *interior rings* for holes
                elif main_outer_outline_poly.equals(cutout_poly):
                    for interior_ring in cutout_poly.interiors:
                        hole_poly_from_interior = Polygon(interior_ring)
                        
                        # Decision for these internal holes is also based on threshold
                        if hole_poly_from_interior.area >= current_cutout_threshold:
                            # Large hole, keep it as a cutout (black dashed outline)
                            x_int, y_int = interior_ring.xy
                            ax.plot(x_int, y_int, color='black', linestyle=':', linewidth=0.8)
                            text_color = 'black'
                        else:
                            # Small hole, it would be filled (red fill)
                            x_int, y_int = interior_ring.xy
                            ax.fill(x_int, y_int, color='red', alpha=0.7)
                            text_color = 'black' # Text color for filled cutouts
                        
                        if hole_poly_from_interior.centroid.is_valid:
                            ax.text(hole_poly_from_interior.centroid.x, hole_poly_from_interior.centroid.y, f"{hole_poly_from_interior.area:.1f}",
                                    fontsize=10, ha='center', va='center', color=text_color, weight='bold',
                                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

            except Exception as e:
                print(f"Error plotting cutout from normalized_outlines: {e}")
                continue


    # Handle legend to show unique labels only
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Ausstecher-Vorschau für {target_x}mm (Schwelle: {current_cutout_threshold:.2f} mm²)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    
    # Calculate dynamic limits for the plot to fit all content
    all_x = []
    all_y = []
    if outer_base_shape_buffered:
        all_x.extend(outer_base_shape_buffered.exterior.xy[0])
        all_y.extend(outer_base_shape_buffered.exterior.xy[1])
    for outline in normalized_outlines:
        all_x.extend(outline.xy[0])
        all_y.extend(outline.xy[1])
    
    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Add a small margin around the content
        margin_x = (max_x - min_x) * 0.1
        margin_y = (max_y - min_y) * 0.1
        
        ax.set_xlim([min_x - margin_x, max_x + margin_x])
        ax.set_ylim([min_y - margin_y, max_y + margin_y])
    else:
        # Fallback to default limits if no content (e.g., empty SVG)
        ax.set_xlim([0, 120])
        ax.set_ylim([0, 120])


    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    try:
        preview_filename = os.path.join(output_path, f"{output_base_name}_cutter_preview_{int(target_x)}mm.png")
        plt.savefig(preview_filename, dpi=200)
    except Exception as e:
        print(f"Error saving 2D preview image: {e}")
        preview_filename = "error_preview.png" # Fallback filename in case of error
    finally:
        plt.close(fig) # Always close the plot to free memory
    
    return os.path.basename(preview_filename)

def generate_stl_preview_image(mesh, output_path, output_base_name, target_x):
    """
    Generates a 2D preview image of the 3D STL mesh using vedo's scene rendering.
    """
    if mesh.is_empty:
        print("Warning: Cannot generate STL preview for an empty mesh.")
        return "empty_mesh_preview.png" # Fallback

    try:
        # Convert trimesh to vedo mesh
        vedo_mesh = vedo.Mesh([mesh.vertices, mesh.faces])

        # Set mesh color to light grey
        vedo_mesh.color('lightgrey') # You can also use [200, 200, 200] for RGB

        # Apply rotation to the vedo mesh for better visibility
        # Rotate 30 degrees around Z-axis (yaw) and -15 degrees around X-axis (pitch)
        vedo_mesh.rotate_z(30)
        vedo_mesh.rotate_x(-15) 

        # Create a plotter in offscreen mode with a specified size (4 times larger)
        plotter = vedo.Plotter(offscreen=True, size=(3200, 3200)) # Increased resolution by factor of 2 again

        # Add the rotated mesh to the plotter
        plotter.add(vedo_mesh)

        # Set camera position for a good isometric view
        # plotter.camera.SetPosition([1, 1, 1]) # This will be relative to the rotated object's bounds
        # plotter.camera.SetFocalPoint([0, 0, 0]) # Center of the scene
        # plotter.camera.SetViewUp([0, 0, 1]) # Z-axis up
        plotter.reset_camera() # Adjust camera to fit the object within the view

        # Take screenshot as a numpy array
        image_data = plotter.screenshot(asarray=True)
        
        # Close the plotter to free resources
        plotter.close()

        # Convert numpy array to PIL Image and save as PNG
        img = Image.fromarray(image_data)

        stl_preview_filename = f"{output_base_name}_stl_preview_{int(target_x)}mm.png"
        full_path = os.path.join(output_path, stl_preview_filename)
        img.save(full_path)
        
        print(f"🖼️ STL preview saved as: {stl_preview_filename}")
        return stl_preview_filename

    except ImportError as e:
        print(f"Error generating STL preview image: {e}. This often means 'vedo' or 'Pillow' is not installed.")
        print("Please ensure you have 'vedo' and 'Pillow' installed using: pip install vedo")
        return "error_stl_preview.png"
    except Exception as e:
        print(f"Error generating STL preview image: {e}")
        # Return a fallback filename if image generation fails
        return "error_stl_preview.png" 


# === Flask-Routen ===
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process_svg', methods=['POST'])
def process_svg():
    if 'svg_file' not in request.files:
        flash('Keine SVG-Datei im Request.')
        return redirect(url_for('index')) # Redirect to index on error
    file = request.files['svg_file']
    if file.filename == '':
        flash('Keine Datei ausgewählt.')
        return redirect(url_for('index')) # Redirect to index on error

    if file and allowed_file(file.filename):
        svg_filename = secure_filename(file.filename)
        svg_filepath = os.path.join(app.config['UPLOAD_FOLDER'], svg_filename)
        file.save(svg_filepath)

        try:
            wall_thickness = float(request.form.get('wall_thickness', 0.75))
            wall_height = float(request.form.get('wall_height', 6.0))
            outer_wall_height = float(request.form.get('outer_wall_height', 3.0))
            base_thickness = float(request.form.get('base_thickness', 2.0))
            base_margin = float(request.form.get('base_margin', 2.0))
            curve_resolution = float(request.form.get('curve_resolution', 0.2))
            internal_bridge_density = float(request.form.get('internal_bridge_density', 0.2))
            num_outer_bridges_per_inner_poly = int(request.form.get('num_outer_bridges_per_inner_poly', 4))
            
            target_x_sizes_str = request.form.get('target_x_sizes', '60, 80, 100')
            target_x_sizes = [float(s.strip()) for s in target_x_sizes_str.split(',') if s.strip()]
            if not target_x_sizes:
                raise ValueError("Keine gültigen Ziel-X-Größen angegeben.")

            user_defined_reference_cutout_threshold = float(request.form.get('cutout_threshold_ref', DEFAULT_REFERENCE_CUTOUT_AREA_THRESHOLD))
            enable_proportional_scaling = request.form.get('enable_proportional_scaling') == 'y'

        except ValueError as e:
            flash(f"Ungültiger Eingabeparameter: {e}")
            return redirect(url_for('index')) # Redirect to index on error
        except Exception as e:
            flash(f"Ein unerwarteter Fehler bei den Formparametern ist aufgetreten: {e}")
            return redirect(url_for('index')) # Redirect to index on error

        generated_files_info = []

        try:
            paths_ref, attributes_ref, svg_attr_ref = svgpathtools.svg2paths2(svg_filepath)
            outlines_ref = []
            vb_min_x_ref, vb_min_y_ref, vb_width_ref, vb_height_ref = 0.0, 0.0, 1.0, 1.0

            svg_width_str_ref = svg_attr_ref.get('width')
            svg_height_str_ref = svg_attr_ref.get('height')
            view_box_ref = svg_attr_ref.get('viewBox')
            global_svg_transform_str_ref = svg_attr_ref.get('transform')

            global_scale_x_ref, global_scale_y_ref = 1.0, 1.0
            global_translate_x_ref, global_translate_y_ref = 0.0, 0.0
            dpi_ref = 96

            if view_box_ref:
                vb_parts_ref = [float(x) for x in view_box_ref.split()]
                vb_min_x_ref, vb_min_y_ref, vb_width_ref, vb_height_ref = vb_parts_ref
                if svg_width_str_ref and svg_height_str_ref:
                    try:
                        parsed_svg_width_ref = parse_svg_length(svg_width_str_ref, dpi=dpi_ref)
                        parsed_svg_height_ref = parse_svg_length(svg_height_str_ref, dpi=dpi_ref)
                    except ValueError as e:
                        print(f"Warnung: SVG-Breite/Höhe '{svg_width_str_ref}', '{svg_height_str_ref}' konnte nicht analysiert werden. Verwende viewBox-Dimensionen als Fallback. Fehler: {e}")
                        parsed_svg_width_ref, parsed_svg_height_ref = vb_width_ref, vb_height_ref
                    if vb_width_ref > 0: global_scale_x_ref = parsed_svg_width_ref / vb_width_ref
                    if vb_height_ref > 0: global_scale_y_ref = parsed_svg_height_ref / vb_height_ref
                    global_translate_x_ref = -vb_min_x_ref * global_scale_x_ref
                    global_translate_y_ref = -vb_min_y_ref * global_scale_y_ref
                else:
                    global_translate_x_ref = -vb_min_x_ref
                    global_translate_y_ref = -vb_min_y_ref
            global_svg_transforms_ref = parse_transform(global_svg_transform_str_ref)

            for path in paths_ref:
                subpaths = path.continuous_subpaths()
                for sub in subpaths:
                    all_points = []
                    first_pt = sub[0].start
                    transformed_x, transformed_y = first_pt.real, first_pt.imag
                    for trans_type, *trans_values in global_svg_transforms_ref:
                        transformed_x, transformed_y = apply_transform(transformed_x, transformed_y, trans_type, trans_values)
                    first_pt_norm_x = transformed_x * global_scale_x_ref + global_translate_x_ref
                    first_pt_norm_y = transformed_y * global_scale_y_ref + global_translate_y_ref
                    all_points.append((first_pt_norm_x, first_pt_norm_y))
                    for segment in sub:
                        sampled = sample_svg_path(segment, resolution=curve_resolution)
                        for pt in sampled[1:]:
                            transformed_x, transformed_y = pt.real, pt.imag
                            for trans_type, *trans_values in global_svg_transforms_ref:
                                transformed_x, transformed_y = apply_transform(transformed_x, transformed_y, trans_type, trans_values)
                            x_norm = transformed_x * global_scale_x_ref + global_translate_x_ref
                            y_norm = transformed_y * global_scale_y_ref + global_translate_y_ref
                            all_points.append((x_norm, y_norm))
                    if not all_points: continue
                    outline = LineString(all_points)
                    outline = affinity.scale(outline, xfact=1, yfact=-1, origin=(0, 0))
                    if not outline.is_valid or len(outline.coords) < 3: continue
                    outlines_ref.append(outline)

            if not outlines_ref:
                flash("Keine gültigen Pfade in der SVG für die Referenzberechnung gefunden. Bitte stellen Sie sicher, dass Ihre SVG korrekt vorbereitet ist.")
                return redirect(url_for('index')) # Redirect to index on error

            combined_outline_ref = unary_union(outlines_ref)
            bounds_ref = combined_outline_ref.bounds
            orig_x_ref = bounds_ref[2] - bounds_ref[0]
            if orig_x_ref < 1e-6:
                flash("Kombinierter SVG-Umriss hat eine Breite von Null oder nahezu Null für die Referenz, kann nicht skaliert werden. Bitte überprüfen Sie Ihre SVG.")
                return redirect(url_for('index')) # Redirect to index on error
            
            for target_x in target_x_sizes:
                current_cutout_threshold = user_defined_reference_cutout_threshold
                if enable_proportional_scaling:
                    current_cutout_threshold = user_defined_reference_cutout_threshold * (target_x / REFERENCE_TARGET_X_SIZE)**2

                reference_outer_base_width = 3.0
                max_cutter_size_for_base_ref = 100.0
                scaled_outer_base_buffer = reference_outer_base_width * (target_x / max_cutter_size_for_base_ref)
                scaled_outer_base_buffer = max(scaled_outer_base_buffer, 0.5)

                paths, attributes, svg_attr = svgpathtools.svg2paths2(svg_filepath)
                outlines = []
                vb_min_x, vb_min_y, vb_width, vb_height = 0.0, 0.0, 1.0, 1.0

                svg_width_str = svg_attr.get('width')
                svg_height_str = svg_attr.get('height')
                view_box = svg_attr.get('viewBox')
                global_svg_transform_str = svg_attr.get('transform')

                global_scale_x, global_scale_y = 1.0, 1.0
                global_translate_x, global_translate_y = 0.0, 0.0
                dpi = 96

                if view_box:
                    vb_parts = [float(x) for x in view_box.split()]
                    vb_min_x, vb_min_y, vb_width, vb_height = vb_parts
                    if svg_width_str and svg_height_str:
                        try:
                            parsed_svg_width = parse_svg_length(svg_width_str, dpi=dpi)
                            parsed_svg_height = parse_svg_length(svg_height_str, dpi=dpi)
                        except ValueError as e:
                            print(f"Warnung: SVG-Breite/Höhe '{svg_width_str}', '{svg_height_str}' konnte nicht analysiert werden. Verwende viewBox-Dimensionen als Fallback. Fehler: {e}")
                            parsed_svg_width, parsed_svg_height = vb_width, vb_height
                        if vb_width > 0: global_scale_x = parsed_svg_width / vb_width
                        if vb_height > 0: global_scale_y = parsed_svg_height / vb_height
                        global_translate_x = -vb_min_x * global_scale_x
                        global_translate_y = -vb_min_y * global_scale_y
                    else:
                        global_translate_x = -vb_min_x
                        global_translate_y = -vb_min_y
                global_svg_transforms = parse_transform(global_svg_transform_str)

                for path in paths:
                    subpaths = path.continuous_subpaths()
                    for sub in subpaths:
                        all_points = []
                        first_pt = sub[0].start
                        transformed_x, transformed_y = first_pt.real, first_pt.imag
                        for trans_type, *trans_values in global_svg_transforms:
                            transformed_x, transformed_y = apply_transform(transformed_x, transformed_y, trans_type, trans_values)
                        first_pt_norm_x = transformed_x * global_scale_x + global_translate_x
                        first_pt_norm_y = transformed_y * global_scale_y + global_translate_y
                        all_points.append((first_pt_norm_x, first_pt_norm_y))
                        for segment in sub:
                            sampled = sample_svg_path(segment, resolution=curve_resolution)
                            for pt in sampled[1:]:
                                transformed_x, transformed_y = pt.real, pt.imag
                                for trans_type, *trans_values in global_svg_transforms:
                                    transformed_x, transformed_y = apply_transform(transformed_x, transformed_y, trans_type, trans_values)
                                x_norm = transformed_x * global_scale_x + global_translate_x
                                y_norm = transformed_y * global_scale_y + global_translate_y
                                all_points.append((x_norm, y_norm))
                        if not all_points: continue
                        outline = LineString(all_points)
                        outline = affinity.scale(outline, xfact=1, yfact=-1, origin=(0, 0))
                        if not outline.is_valid or len(outline.coords) < 3: continue
                        outlines.append(outline)

                if not outlines:
                    flash(f"Keine gültigen Pfade in der SVG für die Zielgröße {target_x}mm gefunden. Bitte stellen Sie sicher, dass Ihre SVG korrekt vorbereitet ist.")
                    return redirect(url_for('index')) # Redirect to index on error

                combined_outline = unary_union(outlines)
                bounds = combined_outline.bounds
                orig_x = bounds[2] - bounds[0]
                if orig_x < 1e-6:
                    flash(f"Kombinierter SVG-Umriss hat eine Breite von Null oder nahezu Null für die Zielgröße {target_x}mm, kann nicht skaliert werden. Bitte überprüfen Sie Ihre SVG.")
                    return redirect(url_for('index')) # Redirect to index on error

                scale_factor = target_x / orig_x
                scaled_outlines = [affinity.scale(ol, xfact=scale_factor, yfact=scale_factor, origin=(0, 0)) for ol in outlines]
                combined_final_outline = unary_union(scaled_outlines)
                min_x_final, min_y_final, max_x_final, max_y_final = combined_final_outline.bounds
                cutting_width = max_x_final - min_x_final
                cutting_height = max_y_final - min_y_final
                normalized_outlines = [affinity.translate(ol, xoff=-min_x_final, yoff=-min_y_final) for ol in scaled_outlines]

                wall_shapes = []
                for outline in normalized_outlines:
                    wall_shape = outline.buffer(wall_thickness / 2.0, cap_style=1, join_style=1, resolution=16)
                    wall_shape = wall_shape.simplify(0.001)
                    wall_shapes.append(wall_shape)

                all_cutouts_before_fill = list_internal_cutouts(normalized_outlines)

                all_removed_hole_geometries = []
                wall_shapes_after_hole_filling = []
                
                for wall_shape in wall_shapes:
                    cleaned_poly, current_removed_holes_geometries = fill_small_cutouts(wall_shape, area_threshold=current_cutout_threshold)
                    all_removed_hole_geometries.extend(current_removed_holes_geometries)
                    wall_shapes_after_hole_filling.append(cleaned_poly)

                max_area_for_tall_wall = -1.0
                if wall_shapes_after_hole_filling:
                    max_area_for_tall_wall = max(poly.area for poly in wall_shapes_after_hole_filling)

                walls_union = unary_union(wall_shapes_after_hole_filling)
                if isinstance(walls_union, MultiPolygon):
                    outer_poly_for_base_and_bridges = max(walls_union.geoms, key=lambda p: p.area)
                else:
                    outer_poly_for_base_and_bridges = walls_union

                wall_meshes = []
                base_meshes = []

                for shape in wall_shapes_after_hole_filling:
                    total_height_from_bed = 0.0
                    if abs(shape.area - max_area_for_tall_wall) < 1e-6:
                        total_height_from_bed = wall_height + outer_wall_height
                    else:
                        total_height_from_bed = wall_height
                    
                    height_for_current_shape = total_height_from_bed - base_thickness
                    height_for_current_shape = max(height_for_current_shape, 0.1) 
                    
                    wall_mesh = extrude_any_shape(shape, height=height_for_current_shape)
                    wall_mesh.apply_translation([0, 0, base_thickness])
                    wall_meshes.append(wall_mesh)

                outer_base_shape_buffered = None
                if wall_shapes_after_hole_filling:
                    outer_base_polygon_candidate = unary_union(wall_shapes_after_hole_filling)
                    if isinstance(outer_base_polygon_candidate, MultiPolygon):
                        outer_base_polygon = max(outer_base_polygon_candidate.geoms, key=lambda p: p.area)
                    else:
                        outer_base_polygon = outer_base_polygon_candidate
                    outer_base_shape_buffered = outer_base_polygon.buffer(scaled_outer_base_buffer, cap_style=1, join_style=1, resolution=16)
                    base_meshes.append(extrude_any_shape(outer_base_shape_buffered, height=base_thickness))

                    for shape in wall_shapes_after_hole_filling:
                        if not shape.equals(outer_base_polygon):
                            inner_base_shape = shape.buffer(base_margin / 2.0, cap_style=1, join_style=1, resolution=16)
                            if not inner_base_shape.is_empty:
                                base_meshes.append(extrude_any_shape(inner_base_shape, height=base_thickness))

                preview_filename = plot_cutter_preview(normalized_outlines, all_removed_hole_geometries, wall_thickness, current_cutout_threshold, target_x, os.path.splitext(svg_filename)[0], outer_base_shape_buffered, app.config['OUTPUT_FOLDER'])

                bridge_width_normal = 8.0 * 0.8
                bridge_width_outer = 3.0

                bridges_internal = generate_bridge_connections(wall_shapes_after_hole_filling, extra_bridge_factor=internal_bridge_density)
                bridges_outer = ensure_minimum_bridges(outer_poly_for_base_and_bridges, wall_shapes_after_hole_filling, num_points_on_inner_perimeter=num_outer_bridges_per_inner_poly)

                bridge_buffers_internal = [line.buffer(bridge_width_normal / 2.0, cap_style=1, join_style=1, resolution=8) for line in bridges_internal]
                bridge_buffers_outer = [line.buffer(bridge_width_outer / 2.0, cap_style=1, join_style=1, resolution=8) for line in bridges_outer]

                bridge_meshes_internal = [extrude_any_shape(b, height=base_thickness) for b in bridge_buffers_internal]
                bridge_meshes_outer = [extrude_any_shape(b, height=base_thickness) for b in bridge_buffers_outer]
                bridge_meshes = bridge_meshes_internal + bridge_meshes_outer

                final_mesh = trimesh.util.concatenate(base_meshes + bridge_meshes + wall_meshes)
                
                final_bounds = final_mesh.bounds
                final_x = final_bounds[1][0] - final_bounds[0][0]
                final_y = final_bounds[1][1] - final_bounds[0][1]
                final_z = final_bounds[1][2] - final_bounds[0][2]

                rounded_cutting_width = int(round(cutting_width, 0))
                rounded_cutting_height = int(round(cutting_height, 0))
                
                output_stl_filename = f"{os.path.splitext(svg_filename)[0]}_{rounded_cutting_width}x{rounded_cutting_height}.stl"
                output_stl_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_stl_filename)
                final_mesh.export(output_stl_filepath)

                # Generate STL preview image
                stl_preview_filename = generate_stl_preview_image(final_mesh, app.config['OUTPUT_FOLDER'], os.path.splitext(svg_filename)[0], target_x)


                generated_files_info.append({
                    'stl_filename': output_stl_filename,
                    'preview_filename': preview_filename, 
                    'stl_preview_filename': stl_preview_filename, # Added STL preview filename
                    'width': rounded_cutting_width,
                    'height': rounded_cutting_height,
                    'total_height': round(final_z, 2)
                })

        except Exception as e:
            flash(f"Ein Fehler ist während der STL-Generierung aufgetreten: {e}")
            if os.path.exists(svg_filepath):
                os.remove(svg_filepath)
            return redirect(url_for('index')) # Redirect to index on error
        finally:
            if os.path.exists(svg_filepath):
                os.remove(svg_filepath)

        return render_template('result.html', generated_files=generated_files_info)
    
    flash('Ungültiger Dateityp. Bitte laden Sie eine SVG-Datei hoch.')
    return redirect(url_for('index')) # Redirect to index on error

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

@app.route('/preview/<filename>')
def serve_preview(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
