import cv2
import argparse
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms


def histogram_matching(source_image, reference_image):
    return match_histograms(source_image, reference_image, channel_axis=-1)


def plot_images(source, reference, matched):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, img, title in zip(axes, [source, reference, matched], ['Source', 'Reference', 'Matched']):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_histograms(source, reference, matched):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, img in enumerate([source, reference, matched]):
        for c, color in enumerate(('red', 'green', 'blue')):
            img_hist, bins = exposure.histogram(img[..., c])
            img_cdf, _ = exposure.cumulative_distribution(img[..., c])
            axes[c, i].plot(bins, img_hist / img_hist.max(), color=color)
            axes[c, i].plot(bins, img_cdf, color='black')
            axes[c, 0].set_ylabel(color.capitalize())
    axes[0, 0].set_title('Source')
    axes[0, 1].set_title('Reference')
    axes[0, 2].set_title('Matched')
    plt.tight_layout()
    plt.show()


def plot_initial_histograms(source, reference):
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for c, color in enumerate(('red', 'green', 'blue')):
        for i, img in enumerate([source, reference]):
            img_hist, bins = exposure.histogram(img[..., c])
            axes[c, i].plot(bins, img_hist / img_hist.max(), color=color)
            axes[c, i].set_title(f'{color.capitalize()} Channel - {"Source" if i == 0 else "Reference"}')
            axes[c, i].set_xlabel('Pixel Value')
            axes[c, i].set_ylabel('Normalized Frequency')
    plt.tight_layout()
    plt.show()


def main(source_path, reference_path):
    source_image = cv2.imread(source_path)
    reference_image = cv2.imread(reference_path)
    matched_image = histogram_matching(source_image, reference_image)

    plot_images(source_image, reference_image, matched_image)
    plot_initial_histograms(source_image, reference_image)
    plot_histograms(source_image, reference_image, matched_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Histogram Matching")
    parser.add_argument("--source_image", required=True, help="Path to the source image")
    parser.add_argument("--reference_image", required=True, help="Path to the reference image")
    args = parser.parse_args()

    main(args.source_image, args.reference_image)
