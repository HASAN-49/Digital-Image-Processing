import cv2
import numpy as np
import matplotlib.pyplot as plt

def linear(x):
    return x.astype(np.float32)

def nonlinear(x):
    x_norm = x / 255.0
    return np.power(x_norm, 0.5) * 255

def none(x):
    return x.astype(np.float32)

def double_threshold(img, low, high, transform_order):
    r, c = img.shape
    out_img = np.zeros_like(img, dtype=np.float32)

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

def simple_threshold(img, threshold):
    bin_img = np.zeros_like(img)
    r, c = img.shape
    for i in range(r):
        for j in range(c):
            if img[i,j] >= threshold:
                bin_img[i,j] = 255
            else:
                bin_img[i,j] = 0
    return bin_img


def display_img(img_set, titles, row, col, k):
    for x in range(len(img_set)):
        plt.subplot(row, col, k)
        plt.imshow(img_set[x], cmap='gray')
        plt.title(titles[x], pad=15)  # Add padding to title to avoid overlap
        plt.axis('off')
        k += 1
        
def display_img1(img_set, titles, row, col, k):
    for i, img_ in enumerate(img_set):
        plt.subplot(2, 4, k)
        plt.hist(img_.ravel(), bins=256, range=(0,256), color='black')
        plt.title(titles[i], pad=15)
        k += 1

def main():
    img_path = '/home/hasan/Desktop/Hasan/7th Semester/CSE4161_2_DIP/Image/nature1.png'  # Change to your image path
    img = cv2.imread(img_path, 0)
    
    img_original = img
    img_type1 = simple_threshold(img, 127)
    img_type2 = double_threshold(img, 80, 170, ['non-linear', 'non-linear', 'linear'])
    img_type3 = double_threshold(img, 80, 170, ['non-linear', 'linear', 'non-linear'])
    
    images = [img_original, img_type1, img_type2, img_type3]
    titles = ['Original Image', 'Type1: Binary Threshold (127)', 'Type2: Non-linear+Non-linear+Linear', 'Type3: Non-linear+Linear+Non-linear']
    
    plt.figure(figsize=(20, 20))
    
    display_img(images, titles, 2, 4, k = 1)
    display_img1(images, titles, 2, 4, k = 5)
    
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()

