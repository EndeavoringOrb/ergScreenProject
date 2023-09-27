"""
row_dict = {
    "time": 1131.8,
    "meters": 5000,
    "avg_split": 113.1,
    "avg_spm": 26,
}

workflow:
number_of_rows = model_1(image)
rows = []
for i in range(number_of_rows):
    row_dict = model_2(image, i)
    rows.append(row_dict)
"""

import csv
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import helper_funcs
import time
from PIL import Image
import numpy as np

def save_dataset(dataset, csv_file = "dataset.csv"):
    """
    dataset = [
        ("image1.jpg", 0, 0.1, 0.2, 0.3, 0.4),
        ("image2.jpg", 1, 0.5, 0.6, 0.7, 0.8),
    ]
    """
    # Write data to CSV
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "row_num", "time", "meters", "avg_split", "avg_spm"])  # Header
        for item in dataset:
            writer.writerow(item)

def load_dataset(csv_file):
    print(f"Loading dataset from {csv_file}...")
    dataset = []
    if not os.path.exists(csv_file):
        return dataset
    # Read data from the CSV file
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        for row in reader:
            dataset.append(row)
    print("Finished loading dataset.")
    return dataset

def label_images(images_folder, current_dataset_path):
    dataset = load_dataset(current_dataset_path)

    print("Finding unlabeled images...")
    # List all image files in the folder
    image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder)]
    # Filter so that already labeled images are not included
    image_paths = [row[0] for row in dataset]
    image_files = [item for item in image_files if item not in image_paths]
    print("Finished.\n")

    number_of_images = len(image_files)

    for image_num, image_filename in enumerate(image_files):
        # Read and display the image
        img = mpimg.imread(image_filename)
        img = np.rot90(img, k=3) # rotate the image so it is upright

        plt.imshow(img)
        plt.axis('off')  # Turn off axis labels and ticks
        plt.show(block=False)

        # Clear Lines
        if image_num != 0:
            helper_funcs.clear_lines(7)

        # Get number of rows from user
        print(f"Image {image_num + 1}/{number_of_images} - ({100 * ((image_num + 1) / number_of_images):.2f}%): {image_filename}")
        row_num = int(input("Enter number of rows in image: "))
        helper_funcs.clear_lines(1)

        # Get each row from user
        for i in range(row_num):
            if i != 0:
                # Clear lines for the next input
                helper_funcs.clear_lines(6)

            # Print which row the user should label
            print(f"Current row: {i + 1}/{row_num}")

            # Pre-print all prompts for better readability
            texts = [
                f"time: ",
                f"meters: ",
                f"avg_split: ",
                f"avg_spm: "
            ]
            texts = [f"{text:>{11}}" for text in texts]
            for text in texts:
                print(text)
            helper_funcs.move_up_lines(len(texts))

            # Ask the user to label the image
            time = helper_funcs.str_to_float_time(input(texts[0]))
            meters = int(input(texts[1]))
            avg_split = helper_funcs.str_to_float_time(input(texts[2]))
            avg_spm = int(input(texts[3]))

            print("Saving...")
            # Save labels to dataset
            dataset.append((image_filename, i, time, meters, avg_split, avg_spm))

        # Save dataset
        save_dataset(dataset, current_dataset_path)

    return dataset

if __name__ == "__main__":
    dataset = label_images("images", "dataset.csv")