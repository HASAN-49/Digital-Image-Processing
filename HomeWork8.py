import cv2
import matplotlib.pyplot as plt

def display_img(img_set, titles, row, col):
    plt.figure(figsize=(12, 10))
    for k in range(len(img_set)):
        plt.subplot(row, col, k + 1)
        plt.imshow(img_set[k], cmap='gray')
        plt.title(titles[k])
        plt.axis('off')
    plt.tight_layout()
    # plt.savefig("images/output/canny_edges_figure.png")
    plt.show()

def main():
    # Load the image in grayscale
    image = cv2.imread("nature1.png", cv2.IMREAD_GRAYSCALE)

    # Apply Canny edge detection with different thresholds
    edges1 = cv2.Canny(image, 50, 150)
    edges2 = cv2.Canny(image, 100, 200)
    edges3 = cv2.Canny(image, 150, 250)

    # Prepare image set and titles
    img_set = [image, edges1, edges2, edges3]
    titles = ["Original", "Canny 50-150", "Canny 100-200", "Canny 150-250"]

    # Display images
    display_img(img_set, titles, row=2, col=2)

if __name__ == "__main__":
    main()
