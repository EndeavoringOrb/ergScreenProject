from mixtureOfExpertsTrain import CustomCNN, interpret_model_output, canny_image, fix_image_orientation
import torch
import csv
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import os
from pillow_heif import register_heif_opener

register_heif_opener()

def test(model_path):
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, device)

    print("Testing model...\n")
    # Read data from the CSV file
    with open("dataset.csv", mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        rows = list(reader)  # Read all rows and convert to a list
        random.shuffle(rows)
        for row in rows:
            image_path, row_num, time, meters, avg_split, avg_spm = row
            transform = transforms.ToTensor()
            image_tensor = transform(canny_image(fix_image_orientation(image_path).resize((768, 768))))
            row_num = int(row_num)
            time = float(time)
            meters = float(meters)
            avg_split = float(avg_split)
            avg_spm = float(avg_spm)

            # show image
            # Read and display the image
            img = np.asarray(image_tensor)

            plt.imshow(img)
            plt.axis('off')  # Turn off axis labels and ticks
            plt.show(block=False)

            model_prediction = model(image_tensor.to(device), torch.tensor([row_num], dtype=torch.float32, device=device)).detach().cpu().numpy()
            model_prediction = interpret_model_output(model_prediction)
            print(f"Row Num: {row_num}")
            print(f"Actual: ({time}, {int(meters)}, {avg_split}, {int(avg_spm)})")
            print(f"Predicted: {model_prediction}")
            input()

def MOEtest(model_path):
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = [path for path in os.listdir(model_path) if path[-2:] == "pt"]
    models = [torch.load(model_path + "/" + path, device) for path in model_paths]

    print("Testing model...\n")
    # Read data from the CSV file
    with open("dataset.csv", mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        rows = list(reader)  # Read all rows and convert to a list
        random.shuffle(rows)
        for row in rows:
            image_path, row_num, time, meters, avg_split, avg_spm = row
            transform = transforms.ToTensor()
            image_tensor = transform(canny_image(fix_image_orientation(image_path).resize((768, 768))))
            image_tensor = image_tensor.unsqueeze(dim=0)
            row_num = int(row_num)
            time = float(time)
            meters = float(meters)
            avg_split = float(avg_split)
            avg_spm = float(avg_spm)

            # show image
            # Read and display the image
            img = np.asarray(image_tensor)[0][0]

            plt.imshow(img)
            plt.axis('off')  # Turn off axis labels and ticks
            plt.show(block=False)

            model_prediction = models[row_num](image_tensor.to(device)).detach().cpu().numpy()
            model_prediction = interpret_model_output(model_prediction)
            print(f"Image: {image_path}")
            print(f"Row Num: {row_num}")
            print(f"Actual: ({time}, {int(meters)}, {avg_split}, {int(avg_spm)})")
            print(f"Predicted: {model_prediction}")
            input()

def predict_image(model_path, image_paths):
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = [path for path in os.listdir(model_path) if path[-2:] == "pt"]
    models = [torch.load(model_path + "/" + path, device) for path in model_paths]

    transform = transforms.ToTensor()

    for image_path in image_paths:
        image_tensor = transform(canny_image(fix_image_orientation(image_path).resize((768, 768))))
        image_tensor = image_tensor.unsqueeze(dim=0)

        # Display the image
        img = np.asarray(image_tensor)[0][0]

        plt.imshow(img)
        plt.axis('off')  # Turn off axis labels and ticks
        plt.show(block=False)

        image_tensor = image_tensor.to(device)

        num_rows = int(input("How many rows of data are there?: "))

        for row_num in range(num_rows):
            with torch.no_grad():
                model_prediction = models[row_num](image_tensor).cpu().numpy()
            model_prediction = interpret_model_output(model_prediction)
            
            print(f"Row Num: {row_num}")
            print(f"Predicted: {model_prediction}\n")
        
        input("Press Enter to go to next image.")
    print("Finished.")

if __name__ == "__main__":
    predict_image("model/model.pt", [f"test_images/{i}" for i in os.listdir("test_images")])