from torchvision import transforms
from PIL import Image
import torch
import csv
import numpy as np

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

def data_generator(model_num, csv_file):
    # Read data from the CSV file
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        for row in reader:
            image_path, row_num, time, meters, avg_split, avg_spm = row
            transform = transforms.ToTensor()
            image = transform(Image.open(image_path).resize((512, 512)))
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
            yield image, torch.tensor([row_num], dtype=torch.float32), torch.tensor(return_x_array, dtype=torch.float32)
            #yield image, torch.tensor([row_num], dtype=torch.float32), torch.tensor([time, meters, avg_split, avg_spm], dtype=torch.float32)

class RowDataFinder(torch.nn.Module):
    def __init__(self):
        super(RowDataFinder, self).__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, 2)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, 2)
        self.conv3 = torch.nn.Conv2d(3, 3, 3, 2)
        self.fc0 = torch.nn.Linear(11908, 256) # +1 for the row_num
        self.fc1 = torch.nn.Linear(256, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 187)

    def forward(self, x, row_num):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(-1)  # Flatten the tensor
        x = torch.cat((x, row_num))
        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        return x.reshape((17, 11))

if __name__ == "__main__":

    # Create an instance of the model
    model = RowDataFinder()

    # Count the number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # Hyperparameters
    learning_rate = 1e-3
    log_interval = 32
    num_epochs = 1_000
    model_save_interval = 1

    # Define loss function and optimizer for model_2
    criterion_model = torch.nn.MSELoss()
    optimizer_model = torch.optim.Adam(model.parameters(), lr=learning_rate)

    steps_per_epoch = 0

    print("Starting training.")
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Initialize running loss
        running_loss = 0.0
        epoch_loss = 0.0

        print("Training Model")
        # Iterate over the data for model
        for i, (image, row_num, targets) in enumerate(data_generator(2, "dataset.csv")):  # Change model_num as needed
            if epoch == 0:
                steps_per_epoch += 1
            optimizer_model.zero_grad()  # Zero the gradients
            outputs = model(image, row_num)  # Forward pass
            loss = criterion_model(outputs, targets)  # Calculate the loss
            loss.backward()  # Backpropagation
            optimizer_model.step()  # Update weights
            running_loss += loss.item()
            epoch_loss += loss.item()

            if (i + 1) % log_interval == 0:
                print(f"Step {i+1}/{'?' if epoch == 0 else steps_per_epoch + 1} - Loss: {running_loss / log_interval:e}")
                running_loss = 0.0

        # Print epoch-level loss for model_2
        print(f"Epoch {epoch+1} Loss: {epoch_loss / (steps_per_epoch + 1):e}")

        if epoch % model_save_interval == 0:
            torch.save(model, "model.pt")

    print("Training finished")