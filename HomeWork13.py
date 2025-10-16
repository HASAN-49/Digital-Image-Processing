import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def adjust_contrast(image, factor):
    mean = np.mean(image)
    adjusted = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    return adjusted

def low_pass_mask(shape, radius):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)
    return mask

def high_pass_mask(shape, radius):
    return 1 - low_pass_mask(shape, radius)

def band_pass_mask(shape, r1, r2):
    return low_pass_mask(shape, r2) - low_pass_mask(shape, r1)

def compute_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    return fshift, magnitude

def apply_filter(fshift, mask):
    filtered = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(filtered))
    return np.abs(img_back)

def display(images, titles, cols=4):
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title, fontsize=9)
        plt.axis('off')
    plt.show()

def process_image(image_path, index):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading: {image_path}")
        return

    results = []
    titles = []

    versions = {
        'Low Contrast': adjust_contrast(img, 0.5),
        'Normal Contrast': img,
        'High Contrast': adjust_contrast(img, 2.0)
    }

    for label, version in versions.items():
        fshift, magnitude = compute_fft(version)
        shape = version.shape

        lp_mask = low_pass_mask(shape, 30)
        hp_mask = high_pass_mask(shape, 30)
        bp_mask = band_pass_mask(shape, 10, 60)

        lp_result = apply_filter(fshift, lp_mask)
        hp_result = apply_filter(fshift, hp_mask)
        bp_result = apply_filter(fshift, bp_mask)

        results.extend([
            version,
            magnitude,
            lp_result,
            hp_result,
            bp_result
        ])
        titles.extend([
            f'Image {index}: {label}',
            f'FFT Spectrum: {label}',
            f'Low-pass: {label}',
            f'High-pass: {label}',
            f'Band-pass: {label}'
        ])

    display(results, titles)

def main():
    image_files = ['nature1.png', 'nature2.png', 'nature_flower.png', 'nature_flower1.png', 'nature_flower2.png']

    for idx, file in enumerate(image_files, 1):
        print(f"\nProcessing {file} ...")
        process_image(file, idx)

if __name__ == "__main__":
    main()
