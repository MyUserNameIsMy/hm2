import cv2
import numpy as np
import argparse


def interpolation(input_image_name, scale_factor):
    image = cv2.imread(input_image_name, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f'Can not open image at path {input_image_name}')
        return

    if scale_factor <= 1:
        print(f'Please run command again with scale factor greater than 1.')
        return

    old_height, old_width = image.shape

    new_height = int(old_height * scale_factor)
    new_width = int(old_width * scale_factor)

    new_image = np.zeros((new_height, new_width), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            pixel_i = int(i / scale_factor)
            pixel_j = int(j / scale_factor)

            new_image[i, j] = image[pixel_i, pixel_j]
    cv2.imshow(f'New image scaled {scale_factor} times', new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('Success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script that interpolates image using Nearest-Neighbor Interpolation technique"
    )

    parser.add_argument("--input_image_name", required=True, type=str, help='Enter input file path.')
    parser.add_argument("--scale_factor",
                        required=True, type=int, help='Enter scale factor integer value greater than 1.')
    args = parser.parse_args()

    interpolation(args.input_image_name, args.scale_factor)
