import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from cornersTrain import CNN

# Define the image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
])

def load_model(model_path, device):
    model = torch.load(model_path, weights_only=False)
    model.eval()
    return model.to(device)

def predict_points(model, image, device):
    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)
    
    # Predict the points
    with torch.no_grad():
        points = model(image).cpu().numpy().flatten()
    
    # Convert points from normalized coordinates to original image size
    points = points.reshape(-1, 2)
    return points

def draw_points_on_image(image, points):
    draw = ImageDraw.Draw(image)
    
    # Draw each point on the image
    for point in points:
        x, y = point
        draw.ellipse((x-25, y-25, x+25, y+25), fill='red', outline='red')
    
    return image

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = load_model('cornerModels/model.pt', device)

    # Load the image
    image_path = "test_images/IMG_E7749.JPG"
    image = Image.open(image_path)

    # Predict the points
    points = predict_points(model, image, device)

    # Convert points from normalized coordinates to original image size
    width, height = image.size
    points[:, 0] *= width  # Scale x coordinates
    points[:, 1] *= height  # Scale y coordinates

    # Draw the points on the image
    image_with_points = draw_points_on_image(image, points)

    # Display the image with points
    plt.imshow(image_with_points)
    plt.axis('off')
    plt.show()

    # Save the image with points
    image_with_points.save('output_image_with_points.jpg')

if __name__ == "__main__":
    main()
