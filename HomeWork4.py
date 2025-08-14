import cv2
import numpy as np
import matplotlib.pyplot as plt

def simple_threshold(img, threshold):
    """One threshold - binary threshold (step function)"""
    bin_img = np.zeros_like(img)
    r, c = img.shape
    for i in range(r):
        for j in range(c):
            if img[i,j] >= threshold:
                bin_img[i,j] = 255
            else:
                bin_img[i,j] = 0
    return bin_img

def double_threshold(img, low, high, transform_order):
    """
    Two thresholds with sequence of transforms.
    transform_order: list of three strings: 'linear', 'non-linear', 'none'
    Applies transforms in order on different regions defined by thresholds.
    Regions:
    - img < low
    - low <= img < high
    - img >= high
    """
    r, c = img.shape
    out_img = np.zeros_like(img, dtype=np.float32)
    
    # Define transform functions:
    def linear(x):
        return x.astype(np.float32)
    def nonlinear(x):
        # Example non-linear transform: gamma correction gamma=0.5 (brighten)
        x_norm = x / 255.0
        return np.power(x_norm, 0.5) * 255
    def none(x):
        return x.astype(np.float32)
    
    transform_map = {
        'linear': linear,
        'non-linear': nonlinear,
        'none': none
    }
    
    for i in range(r):
        for j in range(c):
            val = img[i,j]
            if val < low:
                out_img[i,j] = transform_map[transform_order[0]](np.array([val]))[0]
            elif val < high:
                out_img[i,j] = transform_map[transform_order[1]](np.array([val]))[0]
            else:
                out_img[i,j] = transform_map[transform_order[2]](np.array([val]))[0]
    
    return np.uint8(np.clip(out_img, 0, 255))


def histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    img_eq = cdf_final[img]
    return img_eq

def main():
    img_path = '/home/hasan/Desktop/Hasan/7th Semester/CSE4161_2_DIP/Image/nature1.png'  # Change as needed
    img = cv2.imread(img_path, 0)
    
    # ----------- TYPE 1: One threshold (simple binary) -----------
    th_value = 127
    img_type1 = simple_threshold(img, th_value)
    
    # ----------- TYPE 2: Two thresholds with non-linear + non-linear + linear -----------
    low2, high2 = 80, 170
    # transform order = ['non-linear', 'non-linear', 'linear']
    img_type2 = double_threshold(img, low2, high2, ['non-linear', 'non-linear', 'linear'])
    
    # ----------- TYPE 3: Two thresholds with non-linear + linear + non-linear -----------
    # transform order = ['non-linear', 'linear', 'non-linear']
    img_type3 = double_threshold(img, low2, high2, ['non-linear', 'linear', 'non-linear'])
    
    # ----------- Your two favorite transformations on original -----------
    img_eq = histogram_equalization(img)
    # Gamma correction gamma=0.5 as example nonlinear
    gamma = 0.5
    img_gamma = np.uint8(255 * (img / 255) ** gamma)
    
    # Plotting all images + histograms in one big figure (3 rows, 4 columns)
    plt.figure(figsize=(20, 20))
    
    images = [img, img_type1, img_type2, img_type3, img_eq, img_gamma]
    titles = ['Original Image',
              'Type 1: Binary Threshold (127)',
              'Type 2: Two Thresholds + Non-linear+Non-linear+Linear',
              'Type 3: Two Thresholds + Non-linear+Linear+Non-linear',
              'Histogram Equalization',
              'Gamma Correction (0.5)']
    
    # Show images
    for i in range(len(images)):
        plt.subplot(3, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    
    # Show histograms
    for i in range(len(images)):
        plt.subplot(3, 4, i+1+len(images))
        plt.hist(images[i].ravel(), bins=256, range=(0, 256), color='black')
        plt.title(titles[i] + " Histogram")
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

