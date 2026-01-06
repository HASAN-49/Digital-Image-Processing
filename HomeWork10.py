import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Define custom erosion
def custom_erode(img, kernel):
    m, n = kernel.shape
    pad_h, pad_w = m // 2, n // 2
    padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    out = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i + m, j:j + n]
            if np.array_equal(region[kernel == 1], np.ones(np.sum(kernel)) * 255):
                out[i, j] = 255
    return out


# Define custom dilation
def custom_dilate(img, kernel):
    m, n = kernel.shape
    pad_h, pad_w = m // 2, n // 2
    padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    out = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i + m, j:j + n]
            if np.any(region[kernel == 1] == 255):
                out[i, j] = 255
    return out


# Define custom opening
def custom_open(img, kernel):
    return custom_dilate(custom_erode(img, kernel), kernel)


# Define custom closing
def custom_close(img, kernel):
    return custom_erode(custom_dilate(img, kernel), kernel)


# Define custom top-hat
def custom_tophat(img, kernel):
    return cv2.subtract(img, custom_open(img, kernel))


# Define custom black-hat
def custom_blackhat(img, kernel):
    return cv2.subtract(custom_close(img, kernel), img)


# Define structuring elements
rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
diamond = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=np.uint8,
)

structuring_elements = {
    "rect": rect,
    "ellipse": ellipse,
    "cross": cross,
    "diamond": diamond,
}

# Load and preprocess image
img = cv2.imread("/home/hasan/Desktop/Hasan/7th Semester/CSE4161_2_DIP/Code/original.png", 0)
img_small = cv2.resize(img, (256, 256))
_, img_bin = cv2.threshold(img_small, 127, 255, cv2.THRESH_BINARY)

# Define operations
operations = {
    "Erosion": (cv2.erode, custom_erode),
    "Dilation": (cv2.dilate, custom_dilate),
    "Opening": (lambda im, k: cv2.morphologyEx(im, cv2.MORPH_OPEN, k), custom_open),
    "Closing": (
        lambda im, k: cv2.morphologyEx(im, cv2.MORPH_CLOSE, k),
        custom_close,
    ),
    "Top-Hat": (
        lambda im, k: cv2.morphologyEx(im, cv2.MORPH_TOPHAT, k),
        custom_tophat,
    ),
    "Black-Hat": (
        lambda im, k: cv2.morphologyEx(im, cv2.MORPH_BLACKHAT, k),
        custom_blackhat,
    ),
}

# Create output directory if it doesn't exist
output_dir = "images/output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Perform operations and display results
for se_name, kernel in structuring_elements.items():
    fig, axs = plt.subplots(2, 6, figsize=(15, 5))
    fig.suptitle(
        f"Morphological Operations with {se_name.capitalize()} Structuring Element",
        fontsize=14,
    )

    for idx, (op_name, (opencv_fn, custom_fn)) in enumerate(operations.items()):
        # Apply OpenCV morphological operation
        opencv_res = opencv_fn(img_bin, kernel)
        axs[0, idx].imshow(opencv_res, cmap="gray")
        axs[0, idx].set_title(f"{op_name}\n(OpenCV)", fontsize=8)
        axs[0, idx].axis("off")

        # Apply custom morphological operation
        custom_res = custom_fn(img_bin, kernel)
        axs[1, idx].imshow(custom_res, cmap="gray")
        axs[1, idx].set_title(f"{op_name}\n(Custom)", fontsize=8)
        axs[1, idx].axis("off")

    # Save figure after all operations have been completed
    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    plt.savefig(f"{output_dir}/{se_name}_all.png", dpi=150)

    # Clear the figure to free up memory
    plt.clf()
    plt.close()

print("All operations complete. Results saved to 'images/output/HomwWork10.png'")
