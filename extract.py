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
    
    grid_corners = [top_left, top_right, bottom_left, bottom_right]
    print(f"Grid corners: TL={top_left}, TR={top_right}, BL={bottom_left}, BR={bottom_right}")
else:
    grid_corners = []
    print("No corners found")
    exit()

# Grid dimensions
cols = 8
rows = 13

# Calculate output size (maintain aspect ratio)
cell_size = 50  # pixels per cell
output_width = cols * cell_size
output_height = rows * cell_size

# Source points (detected corners): TL, TR, BL, BR -> need TL, TR, BR, BL order for perspective transform
src_points = np.float32([
    top_left,
    top_right,
    bottom_right,
    bottom_left
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

for row in range(rows):
    for col in range(cols):
        # Calculate cell boundaries
        x1 = col * cell_size + padding
        y1 = row * cell_size + padding
        x2 = (col + 1) * cell_size - padding
        y2 = (row + 1) * cell_size - padding
        
        # Extract the cell
        cell_img = warped[y1:y2, x1:x2]
        
        # Save the cell
        cv2.imwrite(
            os.path.join(output_folder, f"letter_{letter_id:03d}_r{row:02d}_c{col:02d}.png"),
            cell_img
        )
        
        # Draw grid on visualization
        cv2.rectangle(vis_image, 
                      (col * cell_size, row * cell_size), 
                      ((col + 1) * cell_size, (row + 1) * cell_size), 
                      (0, 255, 0), 1)
        cv2.putText(vis_image, str(letter_id), (col * cell_size + 5, row * cell_size + 20),
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