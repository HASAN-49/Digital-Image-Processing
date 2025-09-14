import cv2
import matplotlib.pyplot as plt

def main():
    # Load the image in grayscale
    image = cv2.imread("nature1.png", cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Image not found.")
        return

    # Apply Canny edge detection with different thresholds
    edges1 = cv2.Canny(image, 50, 150)
    edges2 = cv2.Canny(image, 100, 200)
    edges3 = cv2.Canny(image, 150, 250)

    # Create subplots for displaying the images
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(edges1, cmap="gray")
    axes[0, 1].set_title("Canny 50-150")
    axes[1, 0].imshow(edges2, cmap="gray")
    axes[1, 0].set_title("Canny 100-200")
    axes[1, 1].imshow(edges3, cmap="gray")
    axes[1, 1].set_title("Canny 150-250")

    # Hide axes
    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    # plt.savefig("images/output/canny_edges_figure.png")
    plt.show()

if __name__ == "__main__":
    main()

