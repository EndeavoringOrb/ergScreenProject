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
from PIL import Image, ExifTags
import numpy as np
from pillow_heif import register_heif_opener

register_heif_opener()

def remove_dataset_dupes(csv_file, out_file):
    dataset = []
    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        for row in reader:
            dataset.append(tuple(row))
    print(f"dataset length: {len(dataset)}")
    dataset = list(set(dataset))
    print(f"dataset cleaned length: {len(dataset)}")
    dataset = [list(i) for i in dataset]
    dataset = sorted(dataset, key=lambda x: (x[0], x[1]))

    # Write data to CSV
    with open(out_file, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "row_num", "time", "meters", "avg_split", "avg_spm"])  # Header
        for item in dataset:
            writer.writerow(item)

def fix_image_orientation(image_path):
    # Open the image using PIL (Pillow)
    image = Image.open(image_path)

    # Check if the image has EXIF data (metadata)
    if hasattr(image, '_getexif') and image._getexif() is not None:
        # Iterate over EXIF tags and rotate the image if orientation information is found
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                exif = dict(image._getexif().items())
                if orientation in exif:
                    # Rotate the image based on the orientation value
                    if exif[orientation] == 3:
                        image = image.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        image = image.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        image = image.rotate(90, expand=True)

    # Save the fixed image or return it as per your requirement
    # For example, you can save it using image.save('fixed_image.jpg')
    return image

def save_dataset(dataset, csv_file, append):
    """
    dataset = [
        ("image1.jpg", 0, 0.1, 0.2, 0.3, 0.4),
        ("image2.jpg", 1, 0.5, 0.6, 0.7, 0.8),
    ]
    """
    if append:
        # Append data to CSV
        with open(csv_file, mode='a', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            # Check if the file is empty, if so, write the header
            if file.tell() == 0:
                writer.writerow(["image_path", "row_num", "time", "meters", "avg_split", "avg_spm"])
            for item in dataset:
                writer.writerow(item)
    else:
        # Write data to CSV
        with open(csv_file, mode='w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["image_path", "row_num", "time", "meters", "avg_split", "avg_spm"])  # Header
            for item in dataset:
                writer.writerow(item)

def load_dataset(csv_file):
    print(f"Loading dataset from {csv_file}...")
    dataset = []
    if not os.path.exists(csv_file):
        print(f"No dataset exists at {csv_file}. Creating csv file.")
        with open(csv_file, "w", encoding="utf-8") as file:
            pass
        return dataset
    # Read data from the CSV file
    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        for row in reader:
            dataset.append(row)
    print("Finished loading dataset.")
    return dataset

def get_matches(row_data, current_dataset_path):
    # Initialize matches
    matches = []

    # Convert row_data to str format so the == operator works
    row_data = [str(i) for i in row_data]

    # Read data from the CSV file
    with open(current_dataset_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        for row in reader:
            if row[1] == '0':
                if row[2] == row_data[2] and row[3] == row_data[3] and row[4] == row_data[4] and row[5] == row_data[5]:
                    matches.append(row[0])
    return matches

def label_images(images_folder, current_dataset_path):
    old_dataset = load_dataset(current_dataset_path)

    print("Finding unlabeled images...")
    # List all image files in the folder
    image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder)]
    # Filter so that already labeled images are not included
    image_paths = [row[0] for row in old_dataset]
    image_files = [item for item in image_files if item not in image_paths]
    print("Finished.\n")

    number_of_images = len(image_files)

    rows_to_clear = 1

    for image_num, image_filename in enumerate(image_files):
        new_dataset = []
        # Read and display the image
        img = np.asarray(fix_image_orientation(image_filename))

        plt.imshow(img)
        plt.axis('off')  # Turn off axis labels and ticks
        plt.show(block=False)

        # Clear Lines
        if image_num != 0:
            helper_funcs.clear_lines(rows_to_clear + 1)

        # Get number of rows from user
        print(f"Image {image_num + 1}/{number_of_images} - ({100 * ((image_num + 1) / number_of_images):.2f}%): {image_filename}")
        row_num_input = input("Enter number of rows in image: ")
        if row_num_input.lower() == "q":
            rows_to_clear = 1
            os.remove(image_filename)
            continue
        row_num = int(row_num_input)
        helper_funcs.clear_lines(1)

        # Initialize rows to clear tracker
        rows_to_clear = 6

        # Get each row from user
        for i in range(row_num):
            if i != 0:
                # Clear lines for the next input
                helper_funcs.clear_lines(rows_to_clear)

                # Search for matches (for example, two photos of the same erg screen were taken)
                if i == 1:
                    matches = get_matches(new_dataset[-1], current_dataset_path)
                    if len(matches) > 0:
                        print(f"Found Matche(s): ", end="")
                        print(matches[0], end="")
                        for item in matches[1:]:
                            print(f", {item}", end="")
                        print()
                    else:
                        print("No Matches Found.")
                    rows_to_clear = 7
                else:
                    rows_to_clear = 6


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
            new_dataset.append((image_filename, i, time, meters, avg_split, avg_spm))

        # Save dataset
        save_dataset(new_dataset, current_dataset_path, append=True)

if __name__ == "__main__":
    label_images("images", "dataset.csv")
    print("Finished labelling all images. Removing duplicates...")
    remove_dataset_dupes("dataset.csv", "dataset.csv")
    print("Finished.")