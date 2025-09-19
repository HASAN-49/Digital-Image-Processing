import numpy as np
import cv2
import matplotlib.pyplot as plt

def display_images(images, titles):
    plt.figure(figsize=(20, 20))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, len(images), i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')

        plt.subplot(2, len(images), i + 1 + len(images))
        plt.hist(img.ravel(), bins=256, range=(0, 256), color='black')
        plt.title(f"Histogram - {title}")
    plt.tight_layout()
    plt.show()


def histogram_matching_cdf(source, reference):
    src_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])

    src_cdf = src_hist.cumsum() / src_hist.sum()
    ref_cdf = ref_hist.cumsum() / ref_hist.sum()

    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diff = np.abs(ref_cdf - src_cdf[i])
        mapping[i] = np.argmin(diff)

    matched = cv2.LUT(source, mapping)
    return matched


def histogram_matching_spec(source, reference):
    src_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])

    src_hist = src_hist / src_hist.sum()
    ref_hist = ref_hist / ref_hist.sum()

    src_cdf = np.cumsum(src_hist)
    ref_cdf = np.cumsum(ref_hist)

    mapping = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and ref_cdf[j] < src_cdf[i]:
            j += 1
        mapping[i] = j

    matched = cv2.LUT(source, mapping)
    return matched


def adjust_contrast(img, mode='normal'):
    img = img.astype(np.float32)
    if mode == 'low':
        img = img * 0.5 + 64
    elif mode == 'high':
        img = (img - 128) * 2 + 128
    return np.clip(img, 0, 255).astype(np.uint8)


def hist_corr(img1, img2):
    h1, _ = np.histogram(img1.flatten(), 256, [0, 256])
    h2, _ = np.histogram(img2.flatten(), 256, [0, 256])
    return np.corrcoef(h1, h2)[0, 1]


def main():
    img = cv2.imread('nature1.png', cv2.IMREAD_GRAYSCALE)

    source_low = adjust_contrast(img, 'low')
    reference_high = adjust_contrast(img, 'high')

    matched_cdf = histogram_matching_cdf(source_low, reference_high)
    matched_spec = histogram_matching_spec(source_low, reference_high)

    display_images(
        [source_low, reference_high, matched_cdf, matched_spec],
        ['Source (Low)', 'Reference (High)', 'Matched CDF', 'Matched Spec']
    )

    print(f"Histogram Correlation (CDF):   {hist_corr(reference_high, matched_cdf):.4f}")
    print(f"Histogram Correlation (Spec):  {hist_corr(reference_high, matched_spec):.4f}")


if __name__ == "__main__":
    main()
