from torchvision import transforms
from PIL import Image, ImageEnhance, ExifTags, ImageOps
import torch
import csv
import numpy as np
import os
from torchinfo import summary
import json
import cv2
from tqdm import tqdm, trange
from pillow_heif import register_heif_opener
import queue
import threading
from torchvision.models import VisionTransformer

register_heif_opener()

def interpret_model_output(output):
    output = output.squeeze()
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

def canny_image(image):
    # Convert the image to grayscale
    grayscale_image = ImageOps.grayscale(image)

    # Convert PIL image to NumPy array
    grayscale_np = np.array(grayscale_image)

    # Create a cv2 image from the grayscale NumPy array
    cv2_image = cv2.cvtColor(grayscale_np, cv2.COLOR_GRAY2BGR)

    threshold1 = 65
    threshold2 = threshold1 + 25

    # Apply Canny edge detection
    canny_edges = cv2.Canny(cv2_image, threshold1=threshold1, threshold2=threshold2)
    # You can adjust the threshold values according to your image and preference

    # Create a PIL image from the Canny edges numpy array
    canny_image = Image.fromarray(canny_edges)

    return canny_image

def data_generator(csv_file, num_workers, worker_num, save_folder, split="train", split_size=0.8):
    # Read data from the CSV file
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        rows = list(reader)
        split_row = int(len(rows) * split_size)
        if split == "train":
            rows = rows[:split_row]
            image_path_list = [row[0] for row in rows]
            with open(f"{save_folder}/train_dataset_info.json", "w") as f:
                json.dump(image_path_list, f)
        else:
            rows = rows[split_row:]
            image_path_list = [row[0] for row in rows]
            with open(f"{save_folder}/val_dataset_info.json", "w") as f:
                json.dump(image_path_list, f)
        number_of_rows = int(len(rows) / num_workers)
        if worker_num == num_workers - 1:
            rows = rows[worker_num * number_of_rows:]
        else:
            rows = rows[worker_num * number_of_rows:(worker_num + 1) * number_of_rows]
        for row in rows:
            image_path, row_num, time, meters, avg_split, avg_spm = row
            transform = transforms.ToTensor()
            pure_image = fix_image_orientation(image_path).resize((768, 768))
            pure_image = ImageOps.grayscale(pure_image)
            #pure_image = canny_image(pure_image)
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
            for image in image_augmenter(pure_image, augment=False, rotate=True, brightness=False, grayscale=False):
                image = transform(image).to(device)
                image = image.unsqueeze(dim=0)
                yield image, row_num_tensor, return_x_tensor

# Function to add data generator pairs to the queue
def fill_queue(data_queue, num_workers, worker_num, save_folder):
    # Initialize tqdm with the total number of iterations
    for item in enumerate(data_generator("dataset.csv", num_workers, worker_num, save_folder)):
        data_queue.put(item)

class MOERowDataFinder(torch.nn.Module):
    def __init__(self):
        super(MOERowDataFinder, self).__init__()
        #self.vit = VisionTransformer.from_pretrained('vit-base-patch16')
        self.vit = VisionTransformer(768, 16, 12, 4, 128, 3072, num_classes=187)
        self.leaky_relu = torch.nn.LeakyReLU()
        self.fc = torch.nn.Linear(187, 187)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.vit(x)
        x = self.leaky_relu(x)
        x = self.fc(x)
        x = self.sig(x)
        return x.reshape((17,11))
    
class CustomCNN(torch.nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        # Max pooling layer
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Fully connected layers
        self.fc1 = torch.nn.Linear(18432, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 187)  # Output layer with 187 classes
        # Activation funcs
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(self.leaky_relu(self.conv1(x)))
        x = self.pool(self.leaky_relu(self.conv2(x)))
        x = self.pool(self.leaky_relu(self.conv3(x)))
        x = self.pool(self.leaky_relu(self.conv4(x)))
        x = self.pool(self.leaky_relu(self.conv5(x)))
        x = self.pool(self.leaky_relu(self.conv6(x)))
        x = self.pool(self.leaky_relu(self.conv7(x)))
        # Flatten the input for the fully connected layers
        x = x.view((x.shape[0], -1))  # Adjust the input size based on your input dimensions
        # Fully connected layers with ReLU activation
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        # Output layer
        x = self.fc3(x)
        return x.reshape((x.shape[0], 17, 11))

def get_e_notation_of_list(input_list):
    if len(input_list) == 0:
        return ""
    ret_str = f"{input_list[0]:.2e}"
    for item in input_list[1:]:
        ret_str += f", {item:.2e}"
    return f"[{ret_str}]"

if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 2e-5
    log_interval = 50
    num_epochs = 1_000

    # Save and Load Parameters
    save_folder = "MOEmodels"
    load_model = False
    load_folder = "good_models/model1"

    # Generator Parameters
    num_workers = 12

    # Create an instance of the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.")
    if load_model:
        print(f"Loading model from {save_folder}")
        models = [torch.load(f"{load_folder}/rowModel_{i}.pt", device) for i in range(9)]
    else:
        print("Creating model")
        models = [CustomCNN().to(device) for i in range(9)]
    print("Model Summary:")
    summary(models[0], input_data=(torch.rand((1, 1, 768, 768)).to(device)))

    # Define loss function and optimizers
    criterion_model = torch.nn.CrossEntropyLoss()
    optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in models]

    # Make sure save folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Initialize trackers
    if load_model:
        info_dicts = []
        for i in range(9):
            with open(f"{load_folder}/rowModel_{i}_info.json", "r") as f:
                info_dict = json.load(f)
                info_dicts.append(info_dict)
        epoch = max([info_dict["epoch"] for info_dict in info_dicts])
        if save_folder == load_folder:
            lowest_epoch_losses = [info_dict["avg_epoch_val_loss"] for info_dict in info_dicts]
        else:
            lowest_epoch_losses = [1_000_000_000_000_000_000_000] * 9
    else:
        lowest_epoch_losses = [1_000_000_000_000_000_000_000] * 9
        epoch = 0
    steps_per_epoch = 0
    val_steps_per_epoch = 0
    first_epoch = True

    all_data = []
    val_data = []
    step_counts = [0] * 9
    val_step_counts = [0] * 9

    # Training loop
    for _ in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Initialize counters
        epoch_losses = [0] * 9
        epoch_val_losses = [0] * 9
        epoch_avg_losses = [0] * 9
        val_epoch_avg_losses = [0] * 9

        print("Training Model")
        
        if first_epoch:
            # Create a queue to store data generator pairs
            data_queue = queue.Queue(100)

            threads = []

            for worker_num in range(num_workers):
                # Start the thread to fill the queue
                data_thread = threading.Thread(target=fill_queue, args=(data_queue, num_workers, worker_num, save_folder))
                data_thread.start()
                threads.append(data_thread)

            with tqdm(total=None, mininterval=0.5) as pbar:
                while True:
                    try:
                        # Retrieve data from the queue
                        i, (image, row_num, targets) = data_queue.get(timeout=10)
                    except queue.Empty:
                        # If the queue is empty, break the loop
                        break
                    all_data.append((image, row_num, targets))
                    steps_per_epoch += 1
                    row_num = int(row_num.item())
                    step_counts[row_num] += 1
                    optimizers[row_num].zero_grad() # Zero the gradients
                    outputs = models[row_num](image)  # Forward pass
                    loss = criterion_model(outputs.squeeze(), targets)  # Calculate the loss
                    loss.backward()  # Backpropagation
                    optimizers[row_num].step()  # Update weights
                    epoch_losses[row_num] += loss.item()

                    for i in range(9):
                        if step_counts[i] % log_interval == 0 and step_counts[i] > 0:
                            epoch_avg_losses[i] = epoch_losses[i] / step_counts[i]
                    pbar.set_description(f"Losses {get_e_notation_of_list(epoch_avg_losses)}")
                    pbar.update(1)
        else:
            with tqdm(total=len(all_data), mininterval=0.5) as pbar:
                for item in all_data:
                    image, row_num, targets = item
                    row_num = int(row_num.item())
                    optimizers[row_num].zero_grad() # Zero the gradients
                    outputs = models[row_num](image)  # Forward pass
                    loss = criterion_model(outputs.squeeze(), targets)  # Calculate the loss
                    loss.backward()  # Backpropagation
                    optimizers[row_num].step()  # Update weights
                    epoch_losses[row_num] += loss.item()

                    for i in range(9):
                        if step_counts[i] > 0:
                            epoch_avg_losses[i] = epoch_losses[i] / step_counts[i]
                    pbar.set_description(f"Losses {get_e_notation_of_list(epoch_avg_losses)}")
                    pbar.update(1)

        for i in range(9):
            if step_counts[i] % log_interval == 0 and step_counts[i] > 0:
                epoch_avg_losses[i] = epoch_losses[i] / step_counts[i]

        if first_epoch:
            for thread in threads:
                thread.join()

            print("Validating...")
            val_gen = data_generator("dataset.csv", num_workers, worker_num, save_folder, split="val")
        
            with tqdm(total=None, mininterval=0.5) as pbar:
                for image, row_num, targets in val_gen:
                    val_data.append((image, row_num, targets))
                    val_steps_per_epoch += 1
                    row_num = int(row_num.item())
                    val_step_counts[row_num] += 1
                    with torch.no_grad():
                        outputs = models[row_num](image)  # Forward pass
                    loss = criterion_model(outputs.squeeze(), targets)  # Calculate the loss
                    epoch_val_losses[row_num] += loss.item()

                    for i in range(9):
                        if val_step_counts[i] % log_interval == 0 and val_step_counts[i] > 0:
                            val_epoch_avg_losses[i] = epoch_val_losses[i] / val_step_counts[i]
                    pbar.set_description(f"Losses {get_e_notation_of_list(val_epoch_avg_losses)}")
                    pbar.update(1)
        else:
            with tqdm(total=len(val_data), mininterval=0.5) as pbar:
                for item in val_data:
                    image, row_num, targets = item
                    row_num = int(row_num.item())
                    outputs = models[row_num](image)  # Forward pass
                    loss = criterion_model(outputs.squeeze(), targets)  # Calculate the loss
                    epoch_val_losses[row_num] += loss.item()

                    for i in range(9):
                        if val_step_counts[i] > 0:
                            val_epoch_avg_losses[i] = epoch_val_losses[i] / val_step_counts[i]
                    pbar.set_description(f"Losses {get_e_notation_of_list(val_epoch_avg_losses)}")
                    pbar.update(1)

        # Print epoch-level loss for model
        for i in range(9):
            if step_counts[i] > 0:
                epoch_avg_losses[i] = epoch_losses[i] / step_counts[i]
            else:
                epoch_avg_losses[i] = 1e9
        print(f"Epoch {epoch+1} Loss: {get_e_notation_of_list(epoch_avg_losses)}")
        for i in range(9):
            if val_step_counts[i] > 0:
                val_epoch_avg_losses[i] = epoch_val_losses[i] / val_step_counts[i]
            else:
                val_epoch_avg_losses[i] = 1e9
        print(f"Epoch {epoch+1} Val Loss: {get_e_notation_of_list(val_epoch_avg_losses)}")

        for i in range(9):
            if val_epoch_avg_losses[i] <= lowest_epoch_losses[i]:
                lowest_epoch_losses[i] = val_epoch_avg_losses[i]
                torch.save(models[i], f"{save_folder}/rowModel_{i}.pt")
                info_dict = {
                    "epoch": epoch + 1,
                    "avg_epoch_val_loss": lowest_epoch_losses[i],
                    "steps_per_epoch": step_counts[i]
                }
                with open(f"{save_folder}/rowModel_{i}_info.json", "w") as f:
                    json.dump(info_dict, f)
        
        # Increment the epoch counter
        epoch += 1

        # Set first_epoch to false so that we don't count the steps anymore
        first_epoch = False

    print("Training finished")