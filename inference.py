import torch
import torchvision.transforms as transforms
from PIL import Image
from model import UNet
import numpy as np
import matplotlib.pyplot as plt
import os

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

# display the original and post-inferencing iamge
def display_images(image_path, predicted_image):
    original_image = Image.open(image_path)

    # fig
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # og plot
    axes[0].imshow(original_image)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # inference plot
    axes[1].imshow(predicted_image)
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    image_name = os.path.basename(image_path)
    fig.suptitle(image_name, fontsize=14)
    plt.show()

# save the image
def save_image(image_path, predicted_image):
    predict_name = os.path.splitext(os.path.basename(image_path))[0] + '-prediction.png'
    predict_path = os.path.join('predictions', predict_name)

    plt.imshow(predicted_image)
    plt.axis('off')
    plt.savefig(predict_path)

# choose an image folder from within imgages
def enumerate_folders_and_choose():
    subfolders = [f.path for f in os.scandir('images/') if f.is_dir()]
    
    print("Subfolders:")
    for i, folder in enumerate(subfolders):
        print(f"{i+1}. {os.path.basename(folder)}")

    # choose a subfolder
    while True:
        try:
            choice = int(input("Choose a subfolder by typing its number: "))
            if 1 <= choice <= len(subfolders):
                chosen_subfolder = subfolders[choice - 1]
                break
            else:
                print("Invalid choice. Please choose a valid number.")
        except ValueError:
            print("Invalid choice. Please choose a valid number.")
    
    return chosen_subfolder

# choose an image from within that subfolder
def enumerate_images_and_choose(folder_path):
    image_files = os.listdir(folder_path)
    print("Images in the selected subfolder:")
    for i, file in enumerate(image_files):
        print(f"{i+1}. {file}")

    # choose an image
    while True:
        try:
            choice = int(input("Choose an image by typing its number: "))
            if 1 <= choice <= len(image_files):
                chosen_image = os.path.join(folder_path, image_files[choice - 1])
                break
            else:
                print("Invalid choice. Please choose a valid number.")
        except ValueError:
            print("Invalid choice. Please choose a valid number.")

    return chosen_image

if __name__ == '__main__':
    # choose an image and predict
    chosen_subfolder = enumerate_folders_and_choose()
    chosen_image = enumerate_images_and_choose(chosen_subfolder)
    predicted_image = predict_single_image(chosen_image, model)

    # display and save
    display_images(chosen_image, predicted_image)
    save_image(chosen_image, predicted_image)