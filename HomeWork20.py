import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_and_resize(path, width=512):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image {path} not found.")

    h, w = img.shape[:2]
    r = width / float(w)
    dim = (width, int(h * r))

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size : size + 1, -size : size + 1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    return g


def sobel_filters(img):

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = cv2.filter2D(img, -1, Kx)
    Iy = cv2.filter2D(img, -1, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]

                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]

                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]

                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError:
                pass
    return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong


def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                try:
                    if (
                        (img[i + 1, j - 1] == strong)
                        or (img[i + 1, j] == strong)
                        or (img[i + 1, j + 1] == strong)
                        or (img[i, j - 1] == strong)
                        or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong)
                        or (img[i - 1, j] == strong)
                        or (img[i - 1, j + 1] == strong)
                    ):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError:
                    pass
    return img


def custom_canny(img, kernel_size=5, sigma=1.0):
    if kernel_size > 0:
        kernel = gaussian_kernel(kernel_size, sigma)
        smoothed = cv2.filter2D(img, -1, kernel)
    else:
        smoothed = img

    grad_mag, grad_theta = sobel_filters(smoothed)

    non_max = non_max_suppression(grad_mag, grad_theta)

    thresh_img, weak, strong = threshold(non_max, 0.05, 0.15)

    final_img = hysteresis(thresh_img, weak, strong)

    return final_img


image_path = "original.png"

original = load_and_resize(image_path)


no_kernel = custom_canny(original, kernel_size=0)


small_kernel = custom_canny(original, kernel_size=3, sigma=1)


medium_kernel = custom_canny(original, kernel_size=5, sigma=1.4)


large_kernel = custom_canny(original, kernel_size=9, sigma=2.0)


plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
plt.imshow(original, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(no_kernel, cmap="gray")
plt.title("No Gaussian Kernel")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(small_kernel, cmap="gray")
plt.title("Gaussian Kernel (3x3)")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(medium_kernel, cmap="gray")
plt.title("Gaussian Kernel (5x5)")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(large_kernel, cmap="gray")
plt.title("Gaussian Kernel (9x9)")
plt.axis("off")

plt.tight_layout()
plt.savefig("images/output/canny_analysis.png")
print("Experiments complete. Result saved as 'canny_analysis.png'.")
plt.show()