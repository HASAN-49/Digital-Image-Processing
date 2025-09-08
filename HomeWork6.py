import cv2
import numpy as np
import matplotlib.pyplot as plt

def custom_filter2D(img, kernel, mode='same'):
    img_h = img.shape[0]
    img_w = img.shape[1]
    k_h = len(kernel)
    k_w = len(kernel[0])

    flipped_kernel = []
    for i in range(k_h-1, -1, -1):
        row = []
        for j in range(k_w-1, -1, -1):
            row.append(kernel[i][j])
        flipped_kernel.append(row)

    if mode == 'same':
        pad_h = k_h // 2
        pad_w = k_w // 2

        padded_img = np.zeros((img_h + 2*pad_h, img_w + 2*pad_w), dtype=img.dtype)
        padded_img[pad_h:pad_h+img_h, pad_w:pad_w+img_w] = img

    elif mode == 'valid':
        padded_img = img
        pad_h = 0
        pad_w = 0

    out_h = img_h if mode == 'same' else img_h - k_h + 1
    out_w = img_w if mode == 'same' else img_w - k_w + 1

    output = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            val = 0
            for m in range(k_h):
                for n in range(k_w):
                    val += padded_img[i + m, j + n] * flipped_kernel[m][n]
            output[i, j] = val

    return output

def display_img(img_set, titles, row, col):
    for k in range(len(img_set)):
        plt.subplot(row, col, k + 1)
        plt.imshow(img_set[k], cmap='gray')
        plt.title(titles[k])
        plt.axis('off')

def main():
    img_path = '/home/hasan/Desktop/Hasan/7th Semester/CSE4161_2_DIP/Image/nature1.png'  
    img = cv2.imread(img_path, 0)

    kernel_vertical = [[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]]

    kernel_horizontal = [[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]]

    kernel_average = [[1/9, 1/9, 1/9],
                      [1/9, 1/9, 1/9],
                      [1/9, 1/9, 1/9]]

    kernel_random = [[-1, -2, -1],
                     [-3, 6, -3],
                     [1, 2, 1]]

    prewitt_x_kernel = [[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]]

    prewitt_y_kernel = [[-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]]

    laplace_kernel = [[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]]

    filtered_img = [
        img,
        custom_filter2D(img, kernel_vertical, mode='same'),
        custom_filter2D(img, kernel_horizontal, mode='same'),
        custom_filter2D(img, kernel_average, mode='same'),
        custom_filter2D(img, kernel_random, mode='same'),
        custom_filter2D(img, prewitt_x_kernel, mode='same'),
        custom_filter2D(img, prewitt_y_kernel, mode='same'),
        custom_filter2D(img, laplace_kernel, mode='same'),
    ]

    titles = ['Original', 'Vertical Sobel', 'Horizontal Sobel', 'Average Filter', 'Random Filter',
              'Prewitt X', 'Prewitt Y', 'Laplacian']

    plt.figure(figsize=(20, 20))
    display_img(filtered_img, titles, row=3, col=3)
    plt.show()

if __name__ == '__main__':
    main()
