import torch
import torchvision.transforms as transforms
from PIL import Image
from model import UNet
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
])

# load the model
model = UNet(n_channels=3, n_classes=6, bilinear=True).to(device)
model.load_state_dict(torch.load('saved_models/unetSeg.pt'))
model.eval()

# run inference on one image
def predict_single_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    predicted_class = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return predicted_class

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print("Image file not found.")
        sys.exit(1)

    predicted_image = predict_single_image(image_path, model)

    # save the prediction without displaying
    predict_name = os.path.splitext(os.path.basename(image_path))[0] + '-prediction.png'
    predict_path = os.path.join('predictions', predict_name)

    plt.imshow(predicted_image)
    plt.axis('off')
    plt.savefig(predict_path)