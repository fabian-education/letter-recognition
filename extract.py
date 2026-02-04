import cv2
import numpy as np
import os

output_folder = "letters"
image_files = [
    "./data/57-60-bilder-0.jpg",
    "./data/57-60-bilder-1.jpg"
]

# Debug settings
show_grid_preview = True    # Set to False to skip grid visualization
show_grid_detection = True  # Set to False to skip grid line detection debug

# Grid dimensions (inside the detected corners)
inner_cols = 8      # excluding outer border columns
inner_rows = 13     # excluding outer border rows

# Manual extra offset for outer boxes (adjust these values as needed)
extra_left = 50
extra_right = 20
extra_top = 10
extra_bottom = 10

# Output cell size
output_cell_size = 64
padding = 3

os.makedirs(output_folder, exist_ok=True)

# Global letter counter
letter_id = 0

# Store visualization images
vis_images = []
debug_images = []

for image_file in image_files:
    print(f"\nProcessing: {image_file}")
    
    image = cv2.imread(image_file)
    
    if image is None:
        print(f"Warning: Could not read {image_file}, skipping...")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Use binary threshold for cleaner grid detection
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect horizontal lines (use longer kernel)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    # Dilate horizontal lines to close gaps
    horizontal_lines = cv2.dilate(horizontal_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3)), iterations=2)

    # Detect vertical lines (use longer kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    # Dilate vertical lines to close gaps
    vertical_lines = cv2.dilate(vertical_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 20)), iterations=2)

    # Find intersections (where horizontal AND vertical lines meet)
    intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)

    # Dilate intersections to make them easier to detect
    intersections = cv2.dilate(intersections, np.ones((5, 5), np.uint8), iterations=2)

    # Combine grid lines for debug visualization
    grid = cv2.add(horizontal_lines, vertical_lines)

    # Find intersection points
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersections, connectivity=8)

    # Collect intersection centers (skip background label 0)
    intersection_points = []
    for i in range(1, num_labels):
        cx = int(centroids[i][0])
        cy = int(centroids[i][1])
        intersection_points.append((cx, cy))

    print(f"Found {len(intersection_points)} intersections")

    # Find the 4 extreme corners of the grid
    if not intersection_points:
        print("No corners found, skipping...")
        continue
        
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
    
    print(f"Grid corners: TL={top_left}, TR={top_right}, BL={bottom_left}, BR={bottom_right}")

    # Create debug image showing detected grid and corners
    if show_grid_detection:
        grid_debug = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
        
        # Draw all intersection points in white
        for pt in intersection_points:
            cv2.circle(grid_debug, pt, 5, (255, 255, 255), -1)
        
        # Draw the 4 corners with colors
        corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        corner_labels = ["TL", "TR", "BL", "BR"]
        corners = [top_left, top_right, bottom_left, bottom_right]
        
        for i, corner in enumerate(corners):
            cv2.circle(grid_debug, corner, 12, corner_colors[i], -1)
            cv2.putText(grid_debug, corner_labels[i], (corner[0] + 15, corner[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, corner_colors[i], 2)
        
        debug_images.append((os.path.basename(image_file), grid_debug))

    # Calculate cell size based on detected corners
    grid_width = top_right[0] - top_left[0]
    grid_height = bottom_left[1] - top_left[1]
    cell_width = grid_width / inner_cols
    cell_height = grid_height / inner_rows

    print(f"Cell size: {cell_width:.1f} x {cell_height:.1f} px")

    # Expand corners with cell size + extra offset
    offset_tl = (int(top_left[0] - cell_width - extra_left), int(top_left[1] - cell_height - extra_top))
    offset_tr = (int(top_right[0] + cell_width + extra_right), int(top_right[1] - cell_height - extra_top))
    offset_bl = (int(bottom_left[0] - cell_width - extra_left), int(bottom_left[1] + cell_height + extra_bottom))
    offset_br = (int(bottom_right[0] + cell_width + extra_right), int(bottom_right[1] + cell_height + extra_bottom))

    # Calculate output border sizes based on extra offsets
    output_border_left = output_cell_size + int(extra_left * output_cell_size / cell_width)
    output_border_right = output_cell_size + int(extra_right * output_cell_size / cell_width)
    output_border_top = output_cell_size + int(extra_top * output_cell_size / cell_height)
    output_border_bottom = output_cell_size + int(extra_bottom * output_cell_size / cell_height)

    # Total output dimensions
    output_width = output_border_left + (inner_cols * output_cell_size) + output_border_right
    output_height = output_border_top + (inner_rows * output_cell_size) + output_border_bottom

    # Source points (expanded corners)
    src_points = np.float32([
        offset_tl,
        offset_tr,
        offset_br,
        offset_bl
    ])

    # Destination points (rectangle)
    dst_points = np.float32([
        [0, 0],
        [output_width, 0],
        [output_width, output_height],
        [0, output_height]
    ])

    # Get perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Warp the grayscale image
    warped = cv2.warpPerspective(gray, matrix, (output_width, output_height))
    warped_color = cv2.warpPerspective(image, matrix, (output_width, output_height))

    # Create visualization
    vis_image = warped_color.copy()

    # Total grid dimensions (with outer row/column)
    cols = inner_cols + 2
    rows = inner_rows + 2

    start_id = letter_id

    for row in range(rows):
        for col in range(cols):
            # Calculate cell boundaries with variable border sizes
            if col == 0:
                x1 = 0
                x2 = output_border_left
            elif col == cols - 1:
                x1 = output_border_left + (col - 1) * output_cell_size
                x2 = output_width
            else:
                x1 = output_border_left + (col - 1) * output_cell_size
                x2 = x1 + output_cell_size
            
            if row == 0:
                y1 = 0
                y2 = output_border_top
            elif row == rows - 1:
                y1 = output_border_top + (row - 1) * output_cell_size
                y2 = output_height
            else:
                y1 = output_border_top + (row - 1) * output_cell_size
                y2 = y1 + output_cell_size
            
            # Extract the cell with padding
            cell_img = warped[y1 + padding:y2 - padding, x1 + padding:x2 - padding]
            
            # Save the cell
            cv2.imwrite(
                os.path.join(output_folder, f"letter_{letter_id:03d}_r{row:02d}_c{col:02d}.png"),
                cell_img
            )
            
            # Draw grid on visualization
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(vis_image, str(letter_id), (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            letter_id += 1

    print(f"Extracted {letter_id - start_id} letters from this image (IDs {start_id} to {letter_id - 1})")

    # Store visualization for later
    vis_images.append((os.path.basename(image_file), vis_image))

print(f"\nDone. Total {letter_id} letters extracted from {len(image_files)} images.")

# Show all windows at the end
window_offset = 0

if show_grid_detection and debug_images:
    for i, (filename, debug_image) in enumerate(debug_images):
        window_name = f"Grid Detection {i + 1} - {filename}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 400, 700)
        cv2.moveWindow(window_name, window_offset, 50)
        cv2.imshow(window_name, debug_image)
        window_offset += 420

if show_grid_preview and vis_images:
    for i, (filename, vis_image) in enumerate(vis_images):
        window_name = f"Extracted Grid {i + 1} - {filename}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 400, 700)
        cv2.moveWindow(window_name, window_offset, 50)
        cv2.imshow(window_name, vis_image)
        window_offset += 420

if (show_grid_preview and vis_images) or (show_grid_detection and debug_images):
    print("\nPress any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
