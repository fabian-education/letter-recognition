import cv2
import numpy as np
import os

output_folder = "letters"
image = cv2.imread("./data/57-60-bilder-2.jpg")

if image is None:
    raise FileNotFoundError("Image not found.")

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
if intersection_points:
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
else:
    print("No corners found")
    exit()

# Grid dimensions (inside the detected corners)
inner_cols = 8
inner_rows = 13

# Calculate cell size based on detected corners
grid_width = top_right[0] - top_left[0]
grid_height = bottom_left[1] - top_left[1]
cell_width = grid_width / inner_cols
cell_height = grid_height / inner_rows

print(f"Cell size: {cell_width:.1f} x {cell_height:.1f} px")

# Manual extra offset for outer boxes (adjust these values as needed)
extra_left = 50    # extra pixels on left beyond cell_width
extra_right = 20   # extra pixels on right beyond cell_width
extra_top = 10      # extra pixels on top beyond cell_height
extra_bottom = 10   # extra pixels on bottom beyond cell_height

# Expand corners with cell size + extra offset
offset_tl = (int(top_left[0] - cell_width - extra_left), int(top_left[1] - cell_height - extra_top))
offset_tr = (int(top_right[0] + cell_width + extra_right), int(top_right[1] - cell_height - extra_top))
offset_bl = (int(bottom_left[0] - cell_width - extra_left), int(bottom_left[1] + cell_height + extra_bottom))
offset_br = (int(bottom_right[0] + cell_width + extra_right), int(bottom_right[1] + cell_height + extra_bottom))

print(f"Expanded corners: TL={offset_tl}, TR={offset_tr}, BL={offset_bl}, BR={offset_br}")

# Output cell size
output_cell_size = 64  # pixels per cell in output

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

os.makedirs(output_folder, exist_ok=True)

# Create visualization
vis_image = warped_color.copy()

# Extract each cell
padding = 3  # Small padding inside cell
letter_id = 0

# Total grid dimensions (with outer row/column)
cols = inner_cols + 2  # +1 on each side
rows = inner_rows + 2  # +1 on each side

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

print(f"Done. {letter_id} letters extracted ({cols}x{rows} grid).")

# Show warped result with grid
window_name = "Extracted Grid Cells"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 450, 800)
cv2.imshow(window_name, vis_image)
cv2.waitKey(0)
cv2.destroyAllWindows()