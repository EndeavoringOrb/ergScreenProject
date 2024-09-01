import os
import csv
import torch
import numpy as np
from tqdm import tqdm
from PIL import ImageOps
from torchvision import transforms
from mixtureOfExpertsTrain import CustomCNN, interpret_model_output, canny_image, fix_image_orientation

def to_one_hot(num, size=11):
    return [1 if i == num else 0 for i in range(-1, size-1)]

def row_to_classes(row):
    image_path, row_num, time, meters, avg_split, avg_spm = row
    time = float(time)
    meters = float(meters)
    avg_split = float(avg_split)
    avg_spm = float(avg_spm)
    # 17x11
    # 1        2      3    4      5    6
    # time hr, 10min, min, 10sec, sec, dec
    time_hr = time // 3600
    minutes = (time % 3600) // 60
    minutes_10 = minutes // 10
    minutes = (minutes % 10) // 1
    seconds = time % 60
    seconds_10 = seconds // 10
    decimal_seconds = ((seconds % 1) * 10) // 1
    seconds = (seconds % 10) // 1
    if time < 3600:
        time_hr = -1
    if time < 600:
        minutes_10 = -1
    if time < 60:
        minutes = -1
    if time < 10:
        seconds_10 = -1
    if time < 1:
        seconds = -1
    #        7      8     9    10  11
    # meter: 10000, 1000, 100, 10, 1
    meter_10000 = meters // 10_000
    meter_1000 = (meters % 10_000) // 1000
    meter_100 = (meters % 1000) // 100
    meter_10 = (meters % 100) // 10
    meter_1 = (meters % 10) // 1
    if meters < 10_000:
        meter_10000 = -1
    if meters < 1000:
        meter_1000 = -1
    if meters < 100:
        meter_100 = -1
    if meters < 10:
        meter_10 = -1
    #       12   13     14   15
    # split min, 10sec, sec, dec
    split_minutes = (avg_split % 3600) // 60
    split_seconds = avg_split % 60
    split_seconds_10 = split_seconds // 10
    split_decimal_seconds = ((split_seconds % 1) * 10) // 1
    split_seconds = (split_seconds % 10) // 1
    if avg_split < 60:
        split_minutes = -1
    #      16  17
    # spm: 10, 1
    spm_10 = avg_spm // 10
    spm = avg_spm % 10
    return_x_array = [time_hr, minutes_10, minutes, seconds_10, seconds, decimal_seconds, meter_10000, meter_1000, meter_100, meter_10, meter_1, split_minutes, split_seconds_10, split_seconds, split_decimal_seconds, spm_10, spm]
    return return_x_array

class Predictor:
    def __init__(self, model_path) -> None:
        print("Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_paths = [path for path in os.listdir(model_path) if path[-2:] == "pt"]
        self.models = [torch.load(model_path + "/" + path, self.device) for path in model_paths]

    def predict_row(self, row):
        image_path, row_num, time, meters, avg_split, avg_spm = row
        transform = transforms.ToTensor()
        pure_image = fix_image_orientation(image_path).resize((512, 512))
        pure_image = ImageOps.grayscale(pure_image)
        image_tensor = transform(pure_image)
        image_tensor = image_tensor.unsqueeze(dim=0)
        row_num = int(row_num)
        time = float(time)
        meters = float(meters)
        avg_split = float(avg_split)
        avg_spm = float(avg_spm)

        model_prediction = self.models[row_num](image_tensor.to(self.device)).detach().cpu().numpy().squeeze()
        model_prediction = np.asarray([to_one_hot(np.argmax(i)) for i in model_prediction])
        return model_prediction

def get_row_sizes():
    classes = []
    for i in range(9):
        classes.append([])
    with open("dataset.csv", mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        rows = list(reader)  # Read all rows and convert to a list
        rows = rows[:int(len(rows)*0.8)]
        for row in rows:
            row_num = int(row[1])
            x_arr = row_to_classes(row)
            classes[row_num].append(x_arr)
    classes = [np.asarray(i) for i in classes]
    for i in range(len(classes)):
        print(f"# of Row {i+1} Images: {classes[i].shape[0]}")

def get_row_class_sizes(test_row_num):
    class_sizes = np.zeros((17, 11))
    with open("dataset.csv", mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        rows = list(reader)  # Read all rows and convert to a list
        rows = rows[:int(len(rows)*0.8)]
        for row in rows:
            row_num = int(row[1])
            if test_row_num == row_num:
                x_arr = row_to_classes(row)
                x_arr = [to_one_hot(i) for i in x_arr]
                x_arr = np.array(x_arr)
                class_sizes += x_arr
    names = ["time_hr", "minutes_10", "minutes", "seconds_10", "seconds", "decimal_seconds", "meter_10000", "meter_1000", "meter_100", "meter_10", "meter_1", "split_minutes", "split_seconds_10", "split_seconds", "split_decimal_seconds", "spm_10", "spm"]
    for i in range(len(names)):
        print(f"{names[i]:22}: [", end="")
        for j in range(len(class_sizes[i])):
            print(f"{int(class_sizes[i][j]): 4}", end="")
        print("]")

def analyze_row(test_row_num, model_path):
    predictor = Predictor(model_path)
    class_sizes = np.zeros((17, 11))
    accuracies = np.zeros((17, 11))
    valid_rows = 0
    with open("dataset.csv", mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        rows = list(reader)  # Read all rows and convert to a list
        #rows = rows[int(len(rows)*0.9):]
        total_num_rows = len(rows)
        for row in tqdm(rows, desc="Analyzing Rows"):
            row_num = int(row[1])
            if test_row_num == row_num:
                x_arr = row_to_classes(row)
                x_arr = [to_one_hot(i) for i in x_arr]
                x_arr = np.array(x_arr)
                pred_x_arr = predictor.predict_row(row)
                accuracies += np.where(pred_x_arr == x_arr, np.ones_like(x_arr), np.zeros_like(x_arr)) * x_arr
                class_sizes += x_arr
                valid_rows += 1

    print(f"{valid_rows:,} out of {total_num_rows:,} Analyzed rows ({100*(valid_rows/total_num_rows):.2f}%) were of type Row # {test_row_num}")
    for y in range(class_sizes.shape[0]):
        for x in range(class_sizes.shape[1]):
            if class_sizes[y][x] > 0:
                accuracies[y][x] /= class_sizes[y][x]
    names = ["time_hr", "minutes_10", "minutes", "seconds_10", "seconds", "decimal_seconds", "meter_10000", "meter_1000", "meter_100", "meter_10", "meter_1", "split_minutes", "split_seconds_10", "split_seconds", "split_decimal_seconds", "spm_10", "spm"]
    max_str_len = len(str(int(class_sizes.flatten().max())))
    print(f"\nClass Sizes")
    print(f"Class: {' ' * 19}",end="")
    print(f"N/A{' ' * (max(0, max_str_len - 3))}", end="")
    for i in range(10):
        print(f"{i}{' ' * max_str_len}", end="")
    print()
    for i in range(len(names)):
        print(f"{names[i]:22}: [", end="")
        for j in range(len(class_sizes[i])):
            if class_sizes[i][j] > 0:
                color_text = f"\033[38;2;{int(255*(1-accuracies[i][j]))};{int(255*accuracies[i][j])};{0}m"
            else:
                color_text = '\x1b[0m'
            print(f"{color_text}{class_sizes[i][j]: 4}", end="")
        color_text = '\x1b[0m'
        print(f"{color_text}]")

    print(f"\nClass Accuracies")
    print(f"Class: {' ' * 18}",end="")
    print(f"N/A{' '*5}", end="")
    for i in range(10):
        print(f"{i}{' ' * 7}", end="")
    print()
    for i in range(len(names)):
        print(f"{names[i]:22}: [", end="")
        for j in range(len(accuracies[i])):
            if class_sizes[i][j] > 0:
                color_text = f"\033[38;2;{int(255*(1-accuracies[i][j]))};{int(255*accuracies[i][j])};{0}m"
            else:
                color_text = '\x1b[0m'
            text = f"{accuracies[i][j]*100:.2f}%"
            print(f"{color_text}{text:8}", end="")
        color_text = '\x1b[0m'
        print(f"{color_text}]")

if __name__ == "__main__":
    get_row_sizes()
    #get_row_class_sizes(0)
    for i in range(9):
        analyze_row(i, "MOEmodels")