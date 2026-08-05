import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found")
        return None

    image = cv2.resize(image, (128, 128))
    image = image / 255.0

    return image

if __name__ == "__main__":
    sample = preprocess_image("sample.jpg")

    if sample is not None:
        print("Image Preprocessed Successfully")
        print("Shape:", sample.shape)
