import cv2
import numpy as np
import os


def detect_grid_lines(gray_image):
    """Detect horizontal and vertical grid lines in the image."""
    blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
    
    # Use binary threshold for cleaner grid detection
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    horizontal_lines = cv2.dilate(horizontal_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3)), iterations=2)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    vertical_lines = cv2.dilate(vertical_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 20)), iterations=2)
    
    return horizontal_lines, vertical_lines


def find_grid_intersections(horizontal_lines, vertical_lines):
    """Find intersection points where horizontal and vertical lines meet."""
    intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)
    intersections = cv2.dilate(intersections, np.ones((5, 5), np.uint8), iterations=2)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersections, connectivity=8)
    
    intersection_points = []
    for i in range(1, num_labels):  # Skip background label 0
        cx = int(centroids[i][0])
        cy = int(centroids[i][1])
        intersection_points.append((cx, cy))
    
    return intersection_points


def find_grid_corners(intersection_points):
    """Find the 4 extreme corners of the grid from intersection points."""
    if not intersection_points:
        return None
    
    points = np.array(intersection_points)
    
    # Top-left: min x + y
    top_left_idx = np.argmin(points[:, 0] + points[:, 1])
    top_left = tuple(points[top_left_idx])
    
    # Top-right: max x - y
    top_right_idx = np.argmax(points[:, 0] - points[:, 1])
    top_right = tuple(points[top_right_idx])
    
    # Bottom-left: min x - y
    bottom_left_idx = np.argmin(points[:, 0] - points[:, 1])
    bottom_left = tuple(points[bottom_left_idx])
    
    # Bottom-right: max x + y
    bottom_right_idx = np.argmax(points[:, 0] + points[:, 1])
    bottom_right = tuple(points[bottom_right_idx])
    
    return {
        'top_left': top_left,
        'top_right': top_right,
        'bottom_left': bottom_left,
        'bottom_right': bottom_right
    }


def calculate_grid_rotation_angle(corners):
    """Calculate the rotation angle of the grid based on detected corners.
    
    Uses the top edge (top_left to top_right) and left edge to determine
    how much the grid is rotated from horizontal/vertical.
    """
    # Calculate angle from top edge
    dx_top = corners['top_right'][0] - corners['top_left'][0]
    dy_top = corners['top_right'][1] - corners['top_left'][1]
    angle_top = np.degrees(np.arctan2(dy_top, dx_top))
    
    # Calculate angle from bottom edge
    dx_bottom = corners['bottom_right'][0] - corners['bottom_left'][0]
    dy_bottom = corners['bottom_right'][1] - corners['bottom_left'][1]
    angle_bottom = np.degrees(np.arctan2(dy_bottom, dx_bottom))
    
    # Calculate angle from left edge (should be ~90 degrees if grid is straight)
    dx_left = corners['bottom_left'][0] - corners['top_left'][0]
    dy_left = corners['bottom_left'][1] - corners['top_left'][1]
    angle_left = np.degrees(np.arctan2(dy_left, dx_left)) - 90
    
    # Calculate angle from right edge
    dx_right = corners['bottom_right'][0] - corners['top_right'][0]
    dy_right = corners['bottom_right'][1] - corners['top_right'][1]
    angle_right = np.degrees(np.arctan2(dy_right, dx_right)) - 90
    
    # Average the angles for more robust estimation
    avg_angle = (angle_top + angle_bottom + angle_left + angle_right) / 4
    
    return avg_angle


def rotate_image(image, angle, center=None):
    """Rotate image by the given angle around the center."""
    h, w = image.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding box size to avoid cropping
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    # Adjust rotation matrix for the new image size
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2
    
    # Perform rotation with white background
    if len(image.shape) == 2:
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    else:
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    return rotated


def detect_and_correct_rotation(image, gray, angle_threshold=0.3, max_iterations=3):
    """Detect grid rotation and correct it iteratively.
    
    Returns the corrected image and the total rotation applied.
    """
    total_rotation = 0.0
    current_image = image.copy()
    current_gray = gray.copy()
    
    for iteration in range(max_iterations):
        # Detect grid lines
        horizontal_lines, vertical_lines = detect_grid_lines(current_gray)
        intersection_points = find_grid_intersections(horizontal_lines, vertical_lines)
        
        if len(intersection_points) < 4:
            print(f"  Rotation iteration {iteration + 1}: Not enough intersections found")
            break
        
        corners = find_grid_corners(intersection_points)
        if corners is None:
            break
        
        # Calculate rotation angle
        angle = calculate_grid_rotation_angle(corners)
        
        print(f"  Rotation iteration {iteration + 1}: Detected angle = {angle:.3f}°")
        
        # Stop if angle is below threshold
        if abs(angle) < angle_threshold:
            print(f"  Angle below threshold ({angle_threshold}°), stopping rotation correction")
            break
        
        # Rotate to correct
        current_image = rotate_image(current_image, angle)
        current_gray = rotate_image(current_gray, angle)
        total_rotation += angle
    
    return current_image, current_gray, total_rotation


def create_debug_image(horizontal_lines, vertical_lines, intersection_points, corners):
    """Create a debug visualization of detected grid and corners."""
    grid = cv2.add(horizontal_lines, vertical_lines)
    grid_debug = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
    
    # Draw all intersection points in white
    for pt in intersection_points:
        cv2.circle(grid_debug, pt, 5, (255, 255, 255), -1)
    
    # Draw the 4 corners with colors
    corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    corner_labels = ["TL", "TR", "BL", "BR"]
    corner_points = [corners['top_left'], corners['top_right'], corners['bottom_left'], corners['bottom_right']]
    
    for i, corner in enumerate(corner_points):
        cv2.circle(grid_debug, corner, 12, corner_colors[i], -1)
        cv2.putText(grid_debug, corner_labels[i], (corner[0] + 15, corner[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, corner_colors[i], 2)
    
    return grid_debug


def calculate_cell_size(corners, inner_cols, inner_rows):
    """Calculate cell dimensions based on detected corners."""
    grid_width = corners['top_right'][0] - corners['top_left'][0]
    grid_height = corners['bottom_left'][1] - corners['top_left'][1]
    cell_width = grid_width / inner_cols
    cell_height = grid_height / inner_rows
    return cell_width, cell_height


def calculate_expanded_corners(corners, cell_width, cell_height, extra_offsets):
    """Expand corners to include outer border cells."""
    extra_left, extra_right, extra_top, extra_bottom = extra_offsets
    
    offset_tl = (int(corners['top_left'][0] - cell_width - extra_left), 
                 int(corners['top_left'][1] - cell_height - extra_top))
    offset_tr = (int(corners['top_right'][0] + cell_width + extra_right), 
                 int(corners['top_right'][1] - cell_height - extra_top))
    offset_bl = (int(corners['bottom_left'][0] - cell_width - extra_left), 
                 int(corners['bottom_left'][1] + cell_height + extra_bottom))
    offset_br = (int(corners['bottom_right'][0] + cell_width + extra_right), 
                 int(corners['bottom_right'][1] + cell_height + extra_bottom))
    
    return offset_tl, offset_tr, offset_bl, offset_br


def calculate_output_borders(cell_width, cell_height, output_cell_size, extra_offsets):
    """Calculate output border sizes based on extra offsets."""
    extra_left, extra_right, extra_top, extra_bottom = extra_offsets
    
    output_border_left = output_cell_size + int(extra_left * output_cell_size / cell_width)
    output_border_right = output_cell_size + int(extra_right * output_cell_size / cell_width)
    output_border_top = output_cell_size + int(extra_top * output_cell_size / cell_height)
    output_border_bottom = output_cell_size + int(extra_bottom * output_cell_size / cell_height)
    
    return output_border_left, output_border_right, output_border_top, output_border_bottom


def warp_image(image, expanded_corners, output_size):
    """Apply perspective transform to straighten the grid."""
    offset_tl, offset_tr, offset_bl, offset_br = expanded_corners
    output_width, output_height = output_size
    
    src_points = np.float32([offset_tl, offset_tr, offset_br, offset_bl])
    dst_points = np.float32([
        [0, 0],
        [output_width, 0],
        [output_width, output_height],
        [0, output_height]
    ])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, matrix, (output_width, output_height))
    
    return warped


def get_cell_boundaries(row, col, rows, cols, output_cell_size, output_borders, output_size):
    """Calculate cell boundaries for a given row and column."""
    output_border_left, output_border_right, output_border_top, output_border_bottom = output_borders
    output_width, output_height = output_size
    
    if col == 0:
        x1, x2 = 0, output_border_left
    elif col == cols - 1:
        x1 = output_border_left + (col - 1) * output_cell_size
        x2 = output_width
    else:
        x1 = output_border_left + (col - 1) * output_cell_size
        x2 = x1 + output_cell_size
    
    if row == 0:
        y1, y2 = 0, output_border_top
    elif row == rows - 1:
        y1 = output_border_top + (row - 1) * output_cell_size
        y2 = output_height
    else:
        y1 = output_border_top + (row - 1) * output_cell_size
        y2 = y1 + output_cell_size
    
    return x1, y1, x2, y2


def center_character(cell_img, output_size, padding_percent=0.1, boundary_padding=2, char_darkness_threshold=128):
    """Center the character in the cell with padding around the outside.
    
    1. Apply smoothing filter and thresholding
    2. Find the exact bounding box of the character (the 'orange area')
    3. Add small boundary padding around the letter to avoid cutting
    4. Scale the character to fit within the available space (output_size - 2*padding)
    5. Center it with padding on all sides
    
    Args:
        cell_img: Grayscale cell image
        output_size: Size of output image (square)
        padding_percent: Percentage of output size to use as padding around character
        boundary_padding: Extra pixels around detected bounding box
        char_darkness_threshold: Only pixels DARKER than this value (lower value) are detected as character.
                                 Lower threshold = only very dark pixels detected (e.g., 50)
                                 Higher threshold = lighter gray also detected (e.g., 150)
    """
    # Smoothing filter (Gaussian blur)
    smoothed = cv2.GaussianBlur(cell_img, (3, 3), 0)
    # Threshold: pixels darker than char_darkness_threshold are detected as character
    # THRESH_BINARY_INV: pixels <= threshold become 255 (foreground), pixels > threshold become 0
    _, thresh = cv2.threshold(smoothed, char_darkness_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Dilate to expand detection area and capture stroke edges
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # No character found, return resized original
        return cv2.resize(cell_img, (output_size, output_size))
    
    # Get bounding box of all contours combined (exact character bounds)
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Add boundary padding around the letter (expand the bounding box)
    img_h, img_w = cell_img.shape[:2]
    x_new = max(0, x - boundary_padding)
    y_new = max(0, y - boundary_padding)
    x_end = min(img_w, x + w + boundary_padding)
    y_end = min(img_h, y + h + boundary_padding)
    
    # Extract the character region with boundary padding
    char_img = cell_img[y_new:y_end, x_new:x_end]
    
    char_w = x_end - x_new
    char_h = y_end - y_new
    
    if char_w <= 0 or char_h <= 0:
        return cv2.resize(cell_img, (output_size, output_size))
    
    # Calculate padding in pixels
    padding_px = int(output_size * padding_percent)
    
    # Available space for the character (after padding on all sides)
    available_size = output_size - 2 * padding_px
    
    if available_size <= 0:
        available_size = output_size  # Fallback if padding is too large
        padding_px = 0
    
    # Scale character to fit within the available space while maintaining aspect ratio
    scale = min(available_size / char_w, available_size / char_h)
    new_w = int(char_w * scale)
    new_h = int(char_h * scale)
    
    if new_w <= 0 or new_h <= 0:
        return cv2.resize(cell_img, (output_size, output_size))
    
    char_resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create output image (white background)
    output_img = np.ones((output_size, output_size), dtype=np.uint8) * 255
    
    # Calculate position to center the character within the padded area
    x_offset = padding_px + (available_size - new_w) // 2
    y_offset = padding_px + (available_size - new_h) // 2
    
    # Place character in center
    output_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = char_resized
    
    return output_img


def create_grid_mask(output_size, rows, cols, output_cell_size, output_borders, line_thickness=3):
    """Create a grid mask from calculated cell boundaries (the green lines)."""
    output_width, output_height = output_size
    grid_mask = np.zeros((output_height, output_width), dtype=np.uint8)
    
    # Draw grid lines at cell boundaries
    for row in range(rows):
        for col in range(cols):
            x1, y1, x2, y2 = get_cell_boundaries(
                row, col, rows, cols, output_cell_size, output_borders, output_size
            )
            # Draw rectangle edges (same as green rectangles in visualization)
            cv2.rectangle(grid_mask, (x1, y1), (x2, y2), 255, line_thickness)
    
    return grid_mask


def remove_grid_lines_from_cell(cell_img, grid_mask_cell):
    """Remove grid lines from cell using the grid mask."""
    # Create a copy to modify
    cleaned = cell_img.copy()
    
    # Where the grid mask is white (grid lines detected), set pixels to white
    cleaned[grid_mask_cell > 0] = 255
    
    return cleaned


def remove_border_remnants(cell_img, edge_width=5, lightness_threshold=200):
    """Remove border remnants by clearing light gray pixels near cell edges.
    
    Border remnants are typically light gray (close to white but not pure white).
    This function removes pixels that are lighter than the threshold near edges,
    while preserving darker pixels which are the actual letters.
    
    Args:
        cell_img: Grayscale cell image
        edge_width: Width of edge strip to scan (in pixels)
        lightness_threshold: Pixels with value ABOVE this near edges are set to white.
                            Lower value = more aggressive (e.g., 180 removes medium gray)
                            Higher value = less aggressive (e.g., 230 only removes very light gray)
    """
    cleaned = cell_img.copy()
    h, w = cleaned.shape[:2]
    
    if h <= 2 * edge_width or w <= 2 * edge_width:
        return cleaned
    
    # Create edge mask
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:edge_width, :] = True      # Top edge
    edge_mask[h-edge_width:, :] = True    # Bottom edge
    edge_mask[:, :edge_width] = True      # Left edge
    edge_mask[:, w-edge_width:] = True    # Right edge
    
    # Set light pixels in edge regions to white
    # Pixels with value > lightness_threshold are considered "light" (border remnants)
    # This preserves darker pixels which are the actual letter strokes
    cleaned[edge_mask & (cleaned > lightness_threshold)] = 255
    
    return cleaned


def apply_darkness_threshold(cell_img, darkness_threshold=180):
    """Apply soft thresholding: keep dark pixels, set lighter pixels to white.
    
    This is NOT binary thresholding - dark pixels keep their original grayscale value,
    while pixels lighter than the threshold are set to pure white.
    
    Args:
        cell_img: Grayscale cell image
        darkness_threshold: Pixels with value BELOW this keep their original value.
                           Pixels with value >= this are set to white (255).
                           Lower = stricter, only very dark pixels kept (e.g., 100)
                           Higher = more lenient, lighter strokes kept (e.g., 200)
    """
    result = cell_img.copy()
    # Set pixels that are not dark enough to white
    result[result >= darkness_threshold] = 255
    return result


def extract_and_save_cells(warped_gray, warped_color, warped_grid_mask, output_folder, start_letter_id,
                           rows, cols, save_rows, output_cell_size, output_borders, 
                           output_size, padding, center_padding_percent=0.1, boundary_padding=2, char_darkness_threshold=128,
                           enable_centering=False, border_removal_width=5, border_lightness_threshold=200,
                           ink_darkness_threshold=180, start_letter='A'):
    """Extract cells from warped image and save them.
    
    Args:
        start_letter: The letter that the first row corresponds to (e.g., 'A' or 'P')
    """
    vis_image = warped_color.copy()
    letter_id = start_letter_id
    current_letter_ord = ord(start_letter)
    
    for row in range(rows):
        # Calculate the current letter for this row
        current_letter = chr(current_letter_ord + row) if (current_letter_ord + row) <= ord('Z') else None
        
        for col in range(cols):
            x1, y1, x2, y2 = get_cell_boundaries(
                row, col, rows, cols, output_cell_size, output_borders, output_size
            )
            
            if row < save_rows:
                # Extract the full cell (including grid line areas)
                cell_img = warped_gray[y1:y2, x1:x2]
                
                # Extract corresponding grid mask region (same area)
                grid_mask_cell = warped_grid_mask[y1:y2, x1:x2]
                
                # Remove grid lines using the mask
                cleaned_cell = remove_grid_lines_from_cell(cell_img, grid_mask_cell)
                
                # Now crop to inner area (after padding) from the cleaned cell
                inner_x1 = padding
                inner_y1 = padding
                inner_x2 = (x2 - x1) - padding
                inner_y2 = (y2 - y1) - padding
                cleaned_cell = cleaned_cell[inner_y1:inner_y2, inner_x1:inner_x2]
                
                # Apply darkness threshold: keep dark ink pixels, set lighter pixels to white
                cleaned_cell = apply_darkness_threshold(cleaned_cell, ink_darkness_threshold)
                
                # Remove any remaining border remnants near edges
                cleaned_cell = remove_border_remnants(cleaned_cell, border_removal_width, border_lightness_threshold)
                
                # Center the character with padding and smoothing threshold if enabled
                if enable_centering:
                    out_img = center_character(cleaned_cell, output_cell_size, center_padding_percent, boundary_padding, char_darkness_threshold)
                else:
                    out_img = cv2.resize(cleaned_cell, (output_cell_size, output_cell_size))
                
                # Create letter folder and save with letter name
                if current_letter:
                    letter_folder = os.path.join(output_folder, current_letter)
                    os.makedirs(letter_folder, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(letter_folder, f"{current_letter}_{col:02d}.png"),
                        out_img
                    )
                else:
                    # Fallback for rows beyond Z
                    cv2.imwrite(
                        os.path.join(output_folder, f"letter_{letter_id:03d}_r{row:02d}_c{col:02d}.png"),
                        out_img
                    )
                
                # Draw green rectangle for cell boundaries
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                # Draw orange rectangle for inner area (after padding)
                cv2.rectangle(vis_image, (x1 + padding, y1 + padding), (x2 - padding, y2 - padding), (0, 165, 255), 1)
                
                cv2.putText(vis_image, str(letter_id), (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                letter_id += 1
            else:
                # Draw red rectangle for skipped cells
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(vis_image, "X", (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    return vis_image, letter_id


def process_image(image_path, image_rows, inner_cols, inner_rows, extra_offsets, 
                  output_cell_size, padding, output_folder, start_letter_id, 
                  show_grid_detection, center_padding_percent=0.1, boundary_padding=2,
                  rotation_correction=True, rotation_threshold=0.3, char_darkness_threshold=128,
                  enable_centering=False, border_removal_width=5, border_lightness_threshold=200,
                  ink_darkness_threshold=180, grid_line_thickness=3, start_letter='A'):
    """Process a single image and extract letter cells."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read {image_path}, skipping...")
        return None, None, start_letter_id
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply rotation correction if enabled
    if rotation_correction:
        print("  Checking for rotation correction...")
        image, gray, total_rotation = detect_and_correct_rotation(
            image, gray, angle_threshold=rotation_threshold
        )
        if abs(total_rotation) > 0.01:
            print(f"  Applied total rotation correction: {total_rotation:.3f}°")
        else:
            print("  No significant rotation correction needed")
    
    # Detect grid (on potentially rotated image)
    horizontal_lines, vertical_lines = detect_grid_lines(gray)
    intersection_points = find_grid_intersections(horizontal_lines, vertical_lines)
    
    print(f"Found {len(intersection_points)} intersections")
    
    corners = find_grid_corners(intersection_points)
    if corners is None:
        print("No corners found, skipping...")
        return None, None, start_letter_id
    
    print(f"Grid corners: TL={corners['top_left']}, TR={corners['top_right']}, "
          f"BL={corners['bottom_left']}, BR={corners['bottom_right']}")
    
    # Create debug image
    debug_image = None
    if show_grid_detection:
        debug_image = create_debug_image(horizontal_lines, vertical_lines, 
                                          intersection_points, corners)
    
    # Calculate dimensions
    cell_width, cell_height = calculate_cell_size(corners, inner_cols, inner_rows)
    print(f"Cell size: {cell_width:.1f} x {cell_height:.1f} px")
    
    expanded_corners = calculate_expanded_corners(corners, cell_width, cell_height, extra_offsets)
    output_borders = calculate_output_borders(cell_width, cell_height, output_cell_size, extra_offsets)
    
    output_border_left, output_border_right, output_border_top, output_border_bottom = output_borders
    output_width = output_border_left + (inner_cols * output_cell_size) + output_border_right
    output_height = output_border_top + (inner_rows * output_cell_size) + output_border_bottom
    output_size = (output_width, output_height)
    
    # Warp images
    warped_gray = warp_image(gray, expanded_corners, output_size)
    warped_color = warp_image(image, expanded_corners, output_size)
    
    # Extract cells
    cols = inner_cols + 2
    rows = inner_rows + 2
    save_rows = image_rows + 2
    
    # Create grid mask from calculated cell boundaries (the green lines)
    grid_mask = create_grid_mask(output_size, rows, cols, output_cell_size, output_borders, grid_line_thickness)
    
    vis_image, letter_id = extract_and_save_cells(
        warped_gray, warped_color, grid_mask, output_folder, start_letter_id,
        rows, cols, save_rows, output_cell_size, output_borders, output_size, padding,
        center_padding_percent, boundary_padding, char_darkness_threshold, enable_centering,
        border_removal_width, border_lightness_threshold, ink_darkness_threshold, start_letter
    )
    
    print(f"Extracted {letter_id - start_letter_id} letters (IDs {start_letter_id} to {letter_id - 1})")
    
    return debug_image, vis_image, letter_id


def show_results(debug_images, vis_images, show_grid_detection, show_grid_preview, scale_factor=1.0):
    """Display all result windows with 9:16 aspect ratio."""
    window_width = int(270 * scale_factor)   # 9 * 30 * scale
    window_height = int(480 * scale_factor)  # 16 * 30 * scale
    window_offset = 0
    
    if show_grid_detection and debug_images:
        for i, (filename, debug_image) in enumerate(debug_images):
            window_name = f"Grid Detection {i + 1} - {filename}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, window_width, window_height)
            cv2.moveWindow(window_name, window_offset, 50)
            cv2.imshow(window_name, debug_image)
            window_offset += window_width + 20
    
    if show_grid_preview and vis_images:
        for i, (filename, vis_image) in enumerate(vis_images):
            window_name = f"Extracted Grid {i + 1} - {filename}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, window_width, window_height)
            cv2.moveWindow(window_name, window_offset, 50)
            cv2.imshow(window_name, vis_image)
            window_offset += window_width + 20
    
    if (show_grid_preview and vis_images) or (show_grid_detection and debug_images):
        print("\nPress any key to close all windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # Configuration
    output_folder = "letters"
    image_configs = {
        "Part_A": {"path": "./data/57-60-bilder-2.jpg", "rows": None},
        "Part_B": {"path": "./data/57-60-bilder-3.jpg", "rows": 9},
    }
    
    # Debug settings
    show_grid_preview = True
    show_grid_detection = True
    display_scale_factor = 1.7  # Scale factor for 9:16 windows
    
    # Grid dimensions
    inner_cols = 8
    inner_rows = 13
    
    # Manual extra offset for outer boxes
    extra_offsets = (50, 20, 10, 10)  # left, right, top, bottom
    
    # Output settings
    output_cell_size = 64
    padding = 0             # pixels to remove from each side inside cell before centering
    center_padding_percent = 0.02  # x% padding around centered character (added to outside)
    boundary_padding = 2    # pixels of padding around letter bounding box to avoid cutting
    
    # Rotation correction settings
    rotation_correction = True   # Enable automatic rotation correction
    rotation_threshold = 0.01     # Minimum angle (degrees) to trigger rotation correction
    
    # Character detection threshold for centering
    char_darkness_threshold = 180  # Only pixels darker than this are detected as character
                                   # Lower = stricter, only very dark pixels (e.g., 50)
                                   # Higher = more lenient, includes lighter strokes (e.g., 180)

    # Centering option
    enable_centering = True
    
    # Border remnant removal settings
    border_removal_width = 7         # Width of edge strip to scan for border remnants (pixels)
    border_lightness_threshold = 200 # Pixels lighter than this (value > threshold) near edges are removed
                                     # Lower = more aggressive (180 removes medium gray borders)
                                     # Higher = less aggressive (230 only removes very light gray)
    
    # Ink darkness threshold - applied right after cropping
    ink_darkness_threshold = 220     # Pixels darker than this keep their value, lighter become white
                                     # Lower = stricter, only very dark ink kept (e.g., 100)
                                     # Higher = more lenient, lighter strokes kept (e.g., 200)
    
    # Grid line thickness for mask removal
    grid_line_thickness = 3          # Thickness of grid lines to remove (pixels)

    os.makedirs(output_folder, exist_ok=True)

    # Process images
    letter_id = 0
    vis_images = []
    debug_images = []
    
    # Starting letters for each part
    part_start_letters = {
        "Part_A": 'A',  # A to O (15 letters)
        "Part_B": 'P',  # P to Z (11 letters)
    }

    for part_name, config in image_configs.items():
        image_file = config["path"]
        image_rows = config["rows"] if config["rows"] is not None else inner_rows
        start_letter = part_start_letters.get(part_name, 'A')

        print(f"\nProcessing: {image_file} (rows: {image_rows}, starting letter: {start_letter})")

        debug_image, vis_image, letter_id = process_image(
            image_file, image_rows, inner_cols, inner_rows, extra_offsets,
            output_cell_size, padding, output_folder, letter_id, show_grid_detection,
            center_padding_percent, boundary_padding, rotation_correction, rotation_threshold, char_darkness_threshold,
            enable_centering=enable_centering, border_removal_width=border_removal_width,
            border_lightness_threshold=border_lightness_threshold, ink_darkness_threshold=ink_darkness_threshold,
            grid_line_thickness=grid_line_thickness, start_letter=start_letter
        )

        if debug_image is not None:
            debug_images.append((os.path.basename(image_file), debug_image))
        if vis_image is not None:
            vis_images.append((os.path.basename(image_file), vis_image))

    print(f"\nDone. Total {letter_id} letters extracted from {len(image_configs)} images.")

    # Show results
    show_results(debug_images, vis_images, show_grid_detection, show_grid_preview, 
                 display_scale_factor)


if __name__ == "__main__":
    main()
