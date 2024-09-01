from torchvision import transforms
from PIL import Image, ImageEnhance, ExifTags, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if num_workers == 1:
            rows = rows
        elif worker_num == num_workers - 1:
            rows = rows[worker_num * number_of_rows:]
        else:
            rows = rows[worker_num * number_of_rows:(worker_num + 1) * number_of_rows]
        for row in rows:
            image_path, row_num, time, meters, avg_split, avg_spm = row
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
            pure_image = fix_image_orientation(image_path).resize((512, 512))
            pure_image = ImageOps.grayscale(pure_image)
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
    def __init__(self, n_embd):
        super(CustomCNN, self).__init__()
        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        # Max pooling layer
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Fully connected layers
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, n_embd)
        # Activation funcs
        self.leaky_relu = torch.nn.LeakyReLU()
        self.softmax = torch.nn.Softmax(2)

    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(self.leaky_relu(self.conv1(x)))
        x = self.pool(self.leaky_relu(self.conv2(x)))
        x = self.pool(self.leaky_relu(self.conv3(x)))
        # Flatten the input for the fully connected layers
        x = x.view((x.shape[0], -1))  # Adjust the input size based on your input dimensions
        # Fully connected layers with ReLU activation
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        return x

def quantize(x):
    x_bits = torch.round(x / (x.abs().mean()))
    x_bits = torch.clamp(x_bits, -1, 1)
    x_bits = (x_bits - x).detach() + x  # Straight Through Estimator, removes the rounding, absmean, clamp from backprop
    return x_bits

class TernaryLinear(nn.Module):
    """ nn.Linear but with trit quantization from the 1.58 bit paper """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((in_features, out_features)))
        self.norm = nn.LayerNorm((in_features,))
        self.bias = None

    def forward(self, x):
      x = self.norm(x)
      W = quantize(self.weight)
      out = x @ W
      return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, ternary, n_embd, dropout):
        super().__init__()
        if ternary:
          self.key = TernaryLinear(n_embd, head_size, bias=False)
          self.query = TernaryLinear(n_embd, head_size, bias=False)
          self.value = TernaryLinear(n_embd, head_size, bias=False)
        else:
          self.key = nn.Linear(n_embd, head_size, bias=False)
          self.query = nn.Linear(n_embd, head_size, bias=False)
          self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.n_embd = n_embd

    def forward(self, x):
        # x has size (batch, T, channels)
        # output of size (batch, T, head size)
        #B, T, C = x.shape
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)
       
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        #wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, ternary, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, ternary, n_embd, dropout) for _ in range(num_heads)])
        if ternary:
          self.proj = TernaryLinear(head_size * num_heads, n_embd)
        else:
          self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, ternary, dropout):
        super().__init__()
        self.net = nn.Sequential(
            TernaryLinear(n_embd, 4 * n_embd) if ternary else nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            TernaryLinear(4 * n_embd, n_embd) if ternary else nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, head_size, ternary, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        #head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, ternary, n_embd, dropout)
        self.ffwd = FeedFoward(n_embd, ternary, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MyTransformer(nn.Module):

    def __init__(self, ternary, n_embd, dropout, vocab_size, n_head, head_dim, n_layer, n_ctx):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table, embeddings are not ternary
        self.position_embedding_table = nn.Embedding(n_ctx, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, head_dim, ternary, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm

        # out_head is not ternary
        self.out_head1 = nn.Linear(n_embd, n_embd)
        self.out_head2 = nn.Linear(n_embd, vocab_size)
        self.ternary = ternary
        self.n_ctx = n_ctx

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, TernaryLinear) or isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        pos_embeddings = self.position_embedding_table(torch.arange(0, self.n_ctx))
        x = x + pos_embeddings
        x = self.blocks(x) # (B,C)
        x = self.ln_f(x) # (B,C)
        out = self.out_head2(x) # (B,n_embd)
        out = out[:, -1, :].reshape((-1, 1, 17, 11))
        return out

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
    log_interval = 10
    num_epochs = 1_000

    ternary = False
    n_embd = 32
    dropout = 0
    vocab_size = 17*11
    n_head = 8
    head_dim = 32
    n_layer = 1

    patch_size = 32

    n_ctx = 256

    # Save and Load Parameters
    save_folder = "MOEmodels"
    load_model = False
    load_folder = "MOEmodels"

    # Generator Parameters
    num_workers = 12

    # Create an instance of the model
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.")
    if load_model:
        print(f"Loading model from {save_folder}")
        models = [torch.load(f"{load_folder}/rowModel_{i}.pt", device) for i in range(9)]
    else:
        print("Creating model")
        models = [CustomCNN(n_embd).to(device) for _ in range(9)]
        rec_transformers = [MyTransformer(ternary, n_embd, dropout, vocab_size, n_head, head_dim, n_layer, n_ctx) for _ in range(9)]
    print("Model Summary:")
    image = torch.randn(1, 1, 512, 512)
    image_patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).flatten(2, 3) # (B, C, num_patches, patch_size, patch_size)
    summary(models[0], input_data=(image_patches[:, :, 0].to(device)))
    summary(rec_transformers[0], input_data=(torch.rand((1, n_ctx, n_embd)).to(device)))

    # Define loss function and optimizers
    criterion_model = torch.nn.CrossEntropyLoss()
    optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in models]
    rec_transformer_optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in rec_transformers]

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
                    rec_transformer_optimizers[row_num].zero_grad()
                    image_patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).flatten(2, 3).reshape(-1, 1, patch_size, patch_size) # (B, C, num_patches, patch_size, patch_size)
                    image_patch_embeddings = models[row_num](image_patches).unsqueeze(dim=0)
                    outputs = rec_transformers[row_num](image_patch_embeddings)
                    loss = criterion_model(outputs.squeeze(), targets)  # Calculate the loss
                    optimizers[row_num].step()  # Update weights
                    rec_transformer_optimizers[row_num].step()

                    epoch_losses[row_num] += loss.item()

                    for i in range(9):
                        if step_counts[i] % log_interval == 0 and step_counts[i] > 0:
                            epoch_avg_losses[i] = epoch_losses[i] / step_counts[i]
                            pbar.set_description(f"Losses {get_e_notation_of_list(epoch_avg_losses)}")
                    pbar.update(1)
        else:
            tempStepCount = 0
            with tqdm(total=len(all_data), mininterval=0.5) as pbar:
                for item_num, item in enumerate(all_data):
                    image, row_num, targets = item
                    row_num = int(row_num.item())

                    optimizers[row_num].zero_grad() # Zero the gradients
                    rec_transformer_optimizers[row_num].zero_grad()
                    image_patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).flatten(2, 3).reshape(-1, 1, patch_size, patch_size) # (B, C, num_patches, patch_size, patch_size)
                    image_patch_embeddings = models[row_num](image_patches).unsqueeze(dim=0)
                    outputs = rec_transformers[row_num](image_patch_embeddings)
                    loss = criterion_model(outputs.squeeze(), targets)  # Calculate the loss
                    optimizers[row_num].step()  # Update weights
                    rec_transformer_optimizers[row_num].step()

                    epoch_losses[row_num] += loss.item()

                    tempStepCount += 1

                    for i in range(9):
                        if step_counts[i] > 0:
                            epoch_avg_losses[i] = epoch_losses[i] / (item_num + 1)
                    if tempStepCount % log_interval == 0:
                        pbar.set_description(f"Losses {get_e_notation_of_list(epoch_avg_losses)}")
                    pbar.update(1)

        for i in range(9):
            if step_counts[i] % log_interval == 0 and step_counts[i] > 0:
                epoch_avg_losses[i] = epoch_losses[i] / step_counts[i]
        
        print("Validating Model...")

        if first_epoch:
            for thread in threads:
                thread.join()

            val_gen = data_generator("dataset.csv", 1, 0, save_folder, split="val")

            with tqdm(total=None, mininterval=0.5) as pbar:
                for image, row_num, targets in val_gen:
                    val_data.append((image, row_num, targets))
                    val_steps_per_epoch += 1
                    row_num = int(row_num.item())
                    val_step_counts[row_num] += 1
                    with torch.no_grad():
                        image_patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).flatten(2, 3).reshape(-1, 1, patch_size, patch_size) # (B, C, num_patches, patch_size, patch_size)
                        image_patch_embeddings = models[row_num](image_patches).unsqueeze(dim=0)
                        outputs = rec_transformers[row_num](image_patch_embeddings)
                    loss = criterion_model(outputs.squeeze(), targets)  # Calculate the loss
                    epoch_val_losses[row_num] += loss.item()

                    for i in range(9):
                        if val_step_counts[i] % log_interval == 0 and val_step_counts[i] > 0:
                            val_epoch_avg_losses[i] = epoch_val_losses[i] / val_step_counts[i]
                    pbar.set_description(f"Losses {get_e_notation_of_list(val_epoch_avg_losses)}")
                    pbar.update(1)
        else:
            tempStepCount = 0
            with tqdm(total=len(val_data), mininterval=0.5) as pbar:
                for item in val_data:
                    image, row_num, targets = item
                    row_num = int(row_num.item())
                    with torch.no_grad():
                        image_patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).flatten(2, 3).reshape(-1, 1, patch_size, patch_size) # (B, C, num_patches, patch_size, patch_size)
                        image_patch_embeddings = models[row_num](image_patches).unsqueeze(dim=0)
                        outputs = rec_transformers[row_num](image_patch_embeddings)
                    loss = criterion_model(outputs.squeeze(), targets)  # Calculate the loss
                    epoch_val_losses[row_num] += loss.item()

                    tempStepCount += 1

                    for i in range(9):
                        if val_step_counts[i] > 0:
                            val_epoch_avg_losses[i] = epoch_val_losses[i] / val_step_counts[i]
                    if tempStepCount % log_interval == 0:
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