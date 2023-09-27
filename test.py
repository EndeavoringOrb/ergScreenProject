import train
from train import RowDataFinder
import torch
import csv
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

print("Loading model...")
model = torch.load("model.pt")

print("Testing model...\n")
# Read data from the CSV file
with open("dataset.csv", mode='r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header row
    for row in reader:
        image_path, row_num, time, meters, avg_split, avg_spm = row
        transform = transforms.ToTensor()
        image_tensor = transform(Image.open(image_path).resize((512, 512)))
        row_num = int(row_num)
        time = float(time)
        meters = float(meters)
        avg_split = float(avg_split)
        avg_spm = float(avg_spm)
        
        # show image
        # Read and display the image
        img = mpimg.imread(image_path)
        img = np.rot90(img, k=3) # rotate the image so it is upright

        plt.imshow(img)
        plt.axis('off')  # Turn off axis labels and ticks
        plt.show(block=False)

        model_prediction = model(image_tensor, torch.tensor([row_num], dtype=torch.float32)).detach().cpu().numpy()
        model_prediction = train.interpret_model_output(model_prediction)
        print(f"Actual: ({time}, {int(meters)}, {avg_split}, {int(avg_spm)})")
        print(f"Predicted: {model_prediction}")
        input()