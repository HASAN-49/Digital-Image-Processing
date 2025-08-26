import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_img(img_set, color, row, col):
    k = 1
    for x in range(len(img_set)):
        plt.subplot(row, col, k)
        plt.imshow(img_set[x], cmap = 'gray')
        plt.title(color[x])
        plt.axis('off')
        k += 1

def main():
    img_path = '/home/hasan/Desktop/Hasan/7th Semester/CSE4161_2_DIP/Image/nature1.png'
    img = cv2.imread(img_path, 0) # 0 for grayscaling
    
    kernel_vertical = np.array([[-1, 0, 1], # sovel
                        [-2, 0, 2],
                        [-1, 0, 1]])
                        
    kernel_horizontal = np.array([[-1, -2, -1], # sovel
                        [0, 0, 0],
                        [1, 2, 1]])
    
    kernel2 = np.array([[1, 1, 1], 
                        [1, 1, 1],
                        [1, 1, 1]]) / 9
                        
    kernel_random = np.array([[-1, -2, -1], # sovel
                        [-3, 6, -3],
                        [1, 2, 1]])
                        
    # Prewitt filters
    prewitt_x_kernel = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]], dtype=np.float32)
    prewitt_y_kernel = np.array([[-1, -1, -1],
                                 [ 0,  0,  0],
                                 [ 1,  1,  1]], dtype=np.float32)
                                 
    # Laplacian filter
    laplace_kernel = np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=np.float32)
    laplacian = cv2.filter2D(img, -1, laplace_kernel)
    
    new_img_vertical = cv2.filter2D(img, -1, kernel_vertical)
    new_img_horizontal = cv2.filter2D(img, -1, kernel_horizontal)
    new_img2 = cv2.filter2D(img, -1, kernel2)
    new_img_random = cv2.filter2D(img, -1, kernel_random)
    prewitt_x = cv2.filter2D(img, -1, prewitt_x_kernel)
    prewitt_y = cv2.filter2D(img, -1, prewitt_y_kernel)
    
    img_set = [img, new_img_vertical, new_img_horizontal, new_img2, new_img_random, prewitt_x, prewitt_y, laplacian]
    img_title = ['Original', 'Vertical', 'Horizontal', 'simple 1 / 9', 'Random', 'prewitt_x', 'prewitt_x', 'laplacian']
    
    plt.figure(figsize=(20, 20))
    
    display_img(img_set, img_title, row = 3, col = 3)
    
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    main()
