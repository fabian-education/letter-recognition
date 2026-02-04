import cv2
import numpy as np
import os

output_folder = "letters"
image = cv2.imread("./data/57-60-bilder-2.jpg")

min_area = 100      # Lower to catch more letters
max_area = 15000    # Filter out large grid sections
padding = 5
min_width = 8
min_height = 12

if image is None:
    raise FileNotFoundError("Image not found.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Better preprocessing for handwritten text
blur = cv2.GaussianBlur(gray, (3, 3), 0)

# Adaptive threshold works better for uneven lighting/ink
thresh = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 8
)

# Remove grid lines using morphological operations
# Horizontal lines (longer kernel to catch full lines)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)

# Vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

# Remove lines from threshold image
grid_lines = cv2.add(horizontal_lines, vertical_lines)

# Dilate grid lines slightly to ensure complete removal
grid_lines = cv2.dilate(grid_lines, np.ones((3, 3), np.uint8), iterations=1)

thresh_clean = cv2.subtract(thresh, grid_lines)

# Dilate to connect letter parts (dots, broken strokes)
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thresh_clean = cv2.dilate(thresh_clean, dilate_kernel, iterations=1)

# Clean up noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
thresh_clean = cv2.morphologyEx(thresh_clean, cv2.MORPH_CLOSE, kernel)

# Find contours of letters
contours, _ = cv2.findContours(
    thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Sort top to bottom then left to right
bounding_boxes = [cv2.boundingRect(c) for c in contours]
contours = [c for _, c in sorted(
    zip(bounding_boxes, contours),
    key=lambda b: (b[0][1] // 40, b[0][0])  # Group by rows
)]

os.makedirs(output_folder, exist_ok=True)

# Create a copy for visualization
vis_image = image.copy()

letter_id = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Filter by area and dimensions
    if area < min_area or area > max_area:
        continue
    if w < min_width or h < min_height:
        continue
    
    # Skip very wide/flat shapes (likely line remnants)
    aspect_ratio = w / h
    if aspect_ratio > 5 or aspect_ratio < 0.08:
        continue

    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, gray.shape[1])
    y2 = min(y + h + padding, gray.shape[0])

    letter_img = gray[y1:y2, x1:x2]

    cv2.imwrite(
        os.path.join(output_folder, f"letter_{letter_id}.png"),
        letter_img
    )

    # Draw bounding box and label on visualization
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis_image, str(letter_id), (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    letter_id += 1

print(f"Done. {letter_id} letters extracted.")

# Show all detections in one window (scaled to fixed size)
window_name = "Detected Letters"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 720, 1280)
cv2.imshow(window_name, vis_image)
cv2.waitKey(0)
cv2.destroyAllWindows()