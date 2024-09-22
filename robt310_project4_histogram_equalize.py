import cv2
from matplotlib import pyplot as plt
import argparse


def apply_histogram_equalization(input_image_name):
    image = cv2.imread(input_image_name, cv2.IMREAD_GRAYSCALE)

    global_equalized_image = cv2.equalizeHist(image)

    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(40, 40))
    local_equalized_image = clahe.apply(image)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(global_equalized_image, cmap='gray')
    plt.title('Global Histogram Equalization')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(local_equalized_image, cmap='gray')
    plt.title('Local Histogram Equalization (40x40)')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply Global and Local Histogram Equalization to an image.")
    parser.add_argument("--input_image_name", required=True, type=str, help="Enter the input image file path")
    args = parser.parse_args()

    apply_histogram_equalization(args.input_image_name)
