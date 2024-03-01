import torch
import torchvision.transforms as transforms
from PIL import Image
from model import UNet
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
])

# load the model
model = UNet(n_channels=3, n_classes=6, bilinear=True).to(device)
model.load_state_dict(torch.load('saved_models/unet_epoch_7_0.60017.pt'))
model.eval()

# run inference on one image
def predict_single_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    predicted_class = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return predicted_class

# take one pic and inference, could make this take in a user selection
if __name__ == '__main__':
    image_path = 'images/im4.jpg'
    predicted_image = predict_single_image(image_path, model)
    plt.imshow(predicted_image)
    plt.axis('off')
    plt.savefig('images/predicted_mask.png')
    plt.show()
