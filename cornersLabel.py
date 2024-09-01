import cv2
import os
import csv
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
from pillow_heif import register_heif_opener

# Register HEIF opener
register_heif_opener()

# Global variables
points = []
current_image = None
original_image = None
image_path = None
image_width = 0
image_height = 0
scale_factor = 1.0

def mouse_callback(event, x, y, flags, param):
    global points, current_image, image_width, image_height, scale_factor
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            # Convert scaled coordinates back to original image coordinates
            original_x = x / scale_factor
            original_y = y / scale_factor
            # Store points as ratios of original image dimensions
            points.append((original_x / image_width, original_y / image_height))
            cv2.circle(current_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Image", current_image)

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    return folder_path

def get_labeled_images(csv_path):
    labeled_images = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                labeled_images.add(row[0])
    return labeled_images

def load_image(image_path):
    if image_path.lower().endswith('.heic'):
        # Load HEIC image using Pillow
        heif_file = Image.open(image_path)
        image = np.array(heif_file.convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        # Load other image formats using OpenCV
        image = cv2.imread(image_path)
    return image

def scale_image(image, max_width, max_height):
    global scale_factor
    height, width = image.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale_factor = min(scale_w, scale_h, 1)
    
    if scale_factor < 1:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return scaled_image
    return image

# Reorder 4 points into (top-left, top-right, bottom-right, bottom-left)
def reorder_points(points):
    # Convert points to numpy array
    points = np.array(points)
    newPoints = np.zeros_like(points)

    # Sort points based on y coordinate
    sorted_indices_y = np.argsort(points[:, 1])
    points = points[sorted_indices_y]

    # Sort the top two points into (top-left, top-right)
    if points[0][0] > points[1][0]:
        newPoints[0] = points[1]
        newPoints[1] = points[0]
    else:
        newPoints[0] = points[0]
        newPoints[1] = points[1]
    
    # Sort the bottom two points into (bottom-right, bottom-left)
    if points[2][0] > points[3][0]:
        newPoints[2] = points[3]
        newPoints[3] = points[2]
    else:
        newPoints[2] = points[2]
        newPoints[3] = points[3]

    # Return points in the order: top-left, top-right, bottom-right, bottom-left
    return newPoints


def main(csvPath):
    global points, current_image, original_image, image_path, image_width, image_height, scale_factor

    # Select folder containing images
    folder_path = select_folder()
    if not folder_path:
        print("No folder selected. Exiting.")
        return

    # Create or open CSV file for reading and writing
    labeled_images = get_labeled_images(csvPath)

    # Get screen resolution
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # Calculate maximum image display size (80% of screen size)
    max_display_width = int(screen_width * 0.8)
    max_display_height = int(screen_height * 0.8)

    # Open CSV file in append mode
    with open(csvPath, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header if file is empty
        if os.path.getsize(csvPath) == 0:
            csv_writer.writerow(["image_path", "p0_x", "p0_y", "p1_x", "p1_y", "p2_x", "p2_y", "p3_x", "p3_y"])

    # Iterate through images in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.heic')):
            image_path = os.path.join(folder_path, filename)
            
            # Skip if image has already been labeled
            if image_path in labeled_images:
                print(f"Skipping {filename} (already labeled)")
                continue

            original_image = load_image(image_path)
            if original_image is None:
                print(f"Failed to load {filename}. Skipping...")
                continue

            image_height, image_width = original_image.shape[:2]
            current_image = scale_image(original_image, max_display_width, max_display_height)
            points = []

            cv2.namedWindow("Image")
            cv2.setMouseCallback("Image", mouse_callback)

            while True:
                cv2.imshow("Image", current_image)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("Quitting...")
                    return
                elif key == ord('n') or len(points) == 4:
                    if len(points) == 4:
                        # Write to CSV
                        with open(csvPath, 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            points = reorder_points(points)
                            csv_writer.writerow([image_path] + [coord for point in points for coord in point])
                        
                        labeled_images.add(image_path)
                        print(f"Saved labels for {filename}")
                    else:
                        print(f"Skipped {filename}")
                    break

            cv2.destroyAllWindows()

    print("Finished labeling all new images.")

if __name__ == "__main__":
    csvPath = "corners.csv"
    main(csvPath)