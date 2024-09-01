from torchvision import transforms
from PIL import Image, ImageEnhance
import torch
import csv
import numpy as np
import os
from torchinfo import summary
import json
import cv2
from tqdm import tqdm
from pillow_heif import register_heif_opener
import queue
import threading
from joblib import Parallel, delayed

register_heif_opener()

def interpret_model_output(output):
    time_hrs = np.argmax(output[0]) - 1
    time_mins_10 = np.argmax(output[1]) - 1
    time_mins = np.argmax(output[2]) - 1
    time_secs_10 = np.argmax(output[3]) - 1
    time_secs = np.argmax(output[4]) - 1
    time_secs_dec = np.argmax(output[5]) - 1

    time = 0
    time += time_hrs * 3600 if time_hrs != -1 else 0
    time += time_mins_10 * 600 if time_mins_10 != -1 else 0
    time += time_mins * 60 if time_mins != -1 else 0
    time += time_secs_10 * 10 if time_secs_10 != -1 else 0
    time += time_secs if time_secs != -1 else 0
    time += time_secs_dec / 10 if time_secs_dec != -1 else 0

    meters_10000 = np.argmax(output[6]) - 1
    meters_1000 = np.argmax(output[7]) - 1
    meters_100 = np.argmax(output[8]) - 1
    meters_10 = np.argmax(output[9]) - 1
    meters_1 = np.argmax(output[10]) - 1

    meters = 0
    meters += meters_10000 * 10_000 if meters_10000 != -1 else 0
    meters += meters_1000 * 1_000 if meters_1000 != -1 else 0
    meters += meters_100 * 100 if meters_100 != -1 else 0
    meters += meters_10 * 10 if meters_10 != -1 else 0
    meters += meters_1 if meters_1 != -1 else 0

    split_mins = np.argmax(output[11]) - 1
    split_secs_10 = np.argmax(output[12]) - 1
    split_secs = np.argmax(output[13]) - 1
    split_secs_dec = np.argmax(output[14]) - 1

    split = 0
    split += split_mins * 60 if split_mins != -1 else 0
    split += split_secs_10 * 10 if split_secs_10 != -1 else 0
    split += split_secs if split_secs != -1 else 0
    split += split_secs_dec / 10 if split_secs_dec != -1 else 0

    spm_10 = np.argmax(output[15]) - 1
    spm_1 = np.argmax(output[16]) - 1

    spm = 0
    spm += spm_10 * 10 if spm_10 != -1 else 0
    spm += spm_1 if spm_1 != -1 else 0

    return (time, meters, split, spm)

def to_one_hot(num, size=11):
    return [1 if i == num else 0 for i in range(-1, size-1)]

def image_augmenter(image, augment=True, rotate=True, brightness=True, grayscale=True):
    if augment:
        if rotate:
            for i in range(4):
                rotated_image = image.rotate(-90)
                yield rotated_image
                if brightness:
                    brightness_enhancer = ImageEnhance.Brightness(rotated_image)
                    brighter_image = brightness_enhancer.enhance(factor=1.5)
                    yield brighter_image
                    darker_image = brightness_enhancer.enhance(factor=0.5)
                    yield darker_image
        else:
            yield image
            if brightness:
                brightness_enhancer = ImageEnhance.Brightness(image)
                brighter_image = brightness_enhancer.enhance(factor=1.5)
                yield brighter_image
                darker_image = brightness_enhancer.enhance(factor=0.5)
                yield darker_image
        
        if grayscale:
            # gray scale
            # Convert the Pillow image to a NumPy array
            image_np = np.array(image)
            # Convert the NumPy array to grayscale with 3 color channels
            grayscale_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            grayscale_rgb = cv2.cvtColor(grayscale_rgb, cv2.COLOR_GRAY2RGB)
            # Save the resulting image
            grayscale_image = Image.fromarray(grayscale_rgb)
            if rotate:
                for i in range(4):
                    grayscale_image = grayscale_image.rotate(-90)
                    yield grayscale_image
                    if brightness:
                        brightness_enhancer = ImageEnhance.Brightness(grayscale_image)
                        brighter_image = brightness_enhancer.enhance(factor=1.5)
                        yield brighter_image
                        darker_image = brightness_enhancer.enhance(factor=0.5)
                        yield darker_image
            else:
                yield grayscale_image
                if brightness:
                    brightness_enhancer = ImageEnhance.Brightness(grayscale_image)
                    brighter_image = brightness_enhancer.enhance(factor=1.5)
                    yield brighter_image
                    darker_image = brightness_enhancer.enhance(factor=0.5)
                    yield darker_image
    else:
        yield image

def data_generator(csv_file, num_workers, worker_num):
    # Read data from the CSV file
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        rows = list(reader)
        number_of_rows = int(len(rows) / num_workers)
        if worker_num == num_workers - 1:
            rows = rows[worker_num * number_of_rows:]
        else:
            rows = rows[worker_num * number_of_rows:(worker_num + 1) * number_of_rows]
        for row in rows:
            image_path, row_num, time, meters, avg_split, avg_spm = row
            transform = transforms.ToTensor()
            pure_image = Image.open(image_path).resize((512, 512))
            row_num = int(row_num)
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
            return_x_array = [to_one_hot(i) for i in return_x_array]
            
            row_num_tensor = torch.tensor([row_num], dtype=torch.float32, device=device)
            return_x_tensor = torch.tensor(return_x_array, dtype=torch.float32, device=device)
            for image in image_augmenter(pure_image, augment=True, rotate=True, brightness=False, grayscale=False):
                image = transform(image).to(device)
                yield image, row_num_tensor, return_x_tensor

# Function to add data generator pairs to the queue
def fill_queue(data_queue, num_workers, worker_num):
    # Initialize tqdm with the total number of iterations
    for item in enumerate(data_generator("dataset.csv", num_workers, worker_num)):
        data_queue.put(item)

class RowDataFinder(torch.nn.Module):
    def __init__(self):
        super(RowDataFinder, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, 1),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 64, 3, 2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, 1),
            torch.nn.ReLU(),

            torch.nn.Conv2d(128, 128, 3, 2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(128, 128, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3, 1),
            torch.nn.ReLU(),

            torch.nn.Conv2d(256, 256, 3, 2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(256, 256, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, 3, 1),
            torch.nn.ReLU(),

            torch.nn.Conv2d(512, 512, 3, 2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(512, 512, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 17, 3, 1),
            torch.nn.ReLU()
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(9025, 1024),  # +1 for the row_num
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 187),
        )
        self.fc_row_num = torch.nn.Linear(1, 32)

    def forward(self, x, row_num):
        x = self.conv_layers(x)
        x = x.view(-1)  # Flatten the output for fully connected layers
        row_num = self.fc_row_num(row_num)
        x = torch.cat((x, row_num))
        x = self.fc_layers(x)
        x = torch.sigmoid(x)
        return x.reshape((17,11))

if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 2e-5
    log_interval = 50
    num_epochs = 1_000

    # Save and Load Parameters
    save_folder = "model"
    load_model = False

    # Generator Parameters
    num_workers = 12

    # Create an instance of the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.")
    if load_model:
        print(f"Loading model from {save_folder}/model.pt")
        model = torch.load(f"{save_folder}/model.pt", device)
    else:
        print("Creating model")
        model = RowDataFinder().to(device)
    print("Model Summary:")
    summary(model, input_data=(torch.rand((3, 512, 512)).to(device), torch.rand((1)).to(device)))

    # Define loss function and optimizer for model_2
    criterion_model = torch.nn.CrossEntropyLoss()
    optimizer_model = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Make sure save folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Initialize trackers
    if load_model:
        with open(f"{save_folder}/model_info.json", "r") as f:
            info_dict = json.load(f)
        epoch = info_dict["epoch"]
        lowest_epoch_loss = info_dict["avg_epoch_loss"]
    else:
        lowest_epoch_loss = 1_000_000_000_000_000_000_000
        epoch = 0
    steps_per_epoch = 0
    first_epoch = True

    # Training loop
    for _ in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Initialize counters
        epoch_loss = 0.0
        epoch_avg_loss = 0.0
        step_count = 0

        print("Training Model")
        # Create a queue to store data generator pairs
        data_queue = queue.Queue(100)

        threads = []

        for worker_num in range(num_workers):
            # Start the thread to fill the queue
            data_thread = threading.Thread(target=fill_queue, args=(data_queue, num_workers, worker_num))
            data_thread.start()
            threads.append(data_thread)

        with tqdm(total=None if first_epoch else steps_per_epoch, mininterval=0.5) as pbar:
            while True:
                try:
                    # Retrieve data from the queue
                    i, (image, row_num, targets) = data_queue.get(timeout=10)
                except queue.Empty:
                    # If the queue is empty, break the loop
                    break
                if first_epoch:
                    steps_per_epoch += 1
                step_count += 1
                optimizer_model.zero_grad()  # Zero the gradients
                outputs = model(image, row_num)  # Forward pass
                loss = criterion_model(outputs, targets)  # Calculate the loss
                loss.backward()  # Backpropagation
                optimizer_model.step()  # Update weights
                epoch_loss += loss.item()

                if step_count % log_interval == 0:
                    epoch_avg_loss = epoch_loss / step_count
                    pbar.set_description(f"Loss[{epoch_avg_loss:e}]")
                pbar.update(1)

        for thread in threads:
            thread.join()

        # Print epoch-level loss for model
        print(f"Epoch {epoch+1} Loss: {epoch_loss / (steps_per_epoch):e}")

        if epoch_loss / (steps_per_epoch) < lowest_epoch_loss:
            lowest_epoch_loss = epoch_loss / (steps_per_epoch)
            torch.save(model, f"{save_folder}/model.pt")
            info_dict = {
                "epoch": epoch + 1,
                "avg_epoch_loss": lowest_epoch_loss,
                "steps_per_epoch": steps_per_epoch
            }
            with open(f"{save_folder}/model_info.json", "w") as f:
                json.dump(info_dict, f)
        
        # Increment the epoch counter
        epoch += 1

        # Set first_epoch to false so that we don't count the steps anymore
        first_epoch = False

    print("Training finished")