import cv2
import numpy as np
import matplotlib.pyplot as plt

def bit_slicing(image):
    bit_planes = []

    for bit in range(8):
        plane = cv2.bitwise_and(image, 1 << bit)
        plane = np.where(plane > 0, 255, 0)
        bit_planes.append(plane)
    
    return bit_planes

def display_img(bit_planes, row, col):
    fig = plt.subplots(figsize=(20, 20))
    
    k = 1
    for i in range(len(bit_planes)):
        plt.subplot(row, col, k)
        plt.imshow(bit_planes[i], cmap = 'gray')
        plt.title(f'Bit Plane {i}')
        plt.axis('off')
        k += 1

    plt.tight_layout()
    plt.show()
    plt.close()

def main():
    img_path = '/home/hasan/Desktop/Hasan/7th Semester/CSE4161_2_DIP/Image/nature1.png'
    img = cv2.imread(img_path, 0)
    
    bit_planes = bit_slicing(img)
    display_img(bit_planes, row = 2, col = 4)
    
if __name__ == '__main__':
    main()

