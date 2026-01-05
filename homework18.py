import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import pywt
import heapq
from collections import defaultdict


def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def huffman_compression_stats(image):
    flattened = image.flatten()
    freq = defaultdict(int)
    for pix in flattened:
        freq[pix] += 1

    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    huff_dict = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
    code_map = {symbol: code for symbol, code in huff_dict}

    original_bits = len(flattened) * 8
    compressed_bits = 0
    for pix in flattened:
        compressed_bits += len(code_map[pix])

    ratio = original_bits / compressed_bits
    return ratio, compressed_bits


def dct_compress(image, quantization_factor=10):
    h, w = image.shape

    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    img_padded = np.pad(image, ((0, h_pad), (0, w_pad)), "constant")

    Q = (
        np.array(
            [
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ]
        )
        * quantization_factor
        / 10.0
    )

    dct_restored = np.zeros_like(img_padded)

    for i in range(0, img_padded.shape[0], 8):
        for j in range(0, img_padded.shape[1], 8):
            block = img_padded[i : i + 8, j : j + 8] - 128

            dct_block = scipy.fftpack.dct(
                scipy.fftpack.dct(block.T, norm="ortho").T, norm="ortho"
            )

            dct_quant = np.round(dct_block / Q)

            dct_dequant = dct_quant * Q

            idct_block = scipy.fftpack.idct(
                scipy.fftpack.idct(dct_dequant.T, norm="ortho").T, norm="ortho"
            )

            dct_restored[i : i + 8, j : j + 8] = idct_block + 128

    dct_restored = dct_restored[:h, :w]
    return np.clip(dct_restored, 0, 255)


def dwt_compress(image, threshold=20):
    coeffs = pywt.wavedec2(image, "haar", level=2)

    coeffs_thresh = list(coeffs)

    for i in range(1, len(coeffs)):
        coeffs_thresh[i] = tuple(
            map(lambda x: pywt.threshold(x, threshold, mode="hard"), coeffs[i])
        )

    dwt_restored = pywt.waverec2(coeffs_thresh, "haar")

    dwt_restored = dwt_restored[: image.shape[0], : image.shape[1]]

    return np.clip(dwt_restored, 0, 255)


img = cv2.imread("images/birds.jpg", 0)
if img is None:
    img = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(img, (256, 256), 100, 200, -1)
    img = img + np.random.normal(0, 10, img.shape).astype(np.uint8)


huff_ratio, huff_bits = huffman_compression_stats(img)
print(f"Huffman Compression Ratio: {huff_ratio:.2f}")


dct_img = dct_compress(img, quantization_factor=20)
dct_psnr = calculate_psnr(img, dct_img)
print(f"DCT PSNR: {dct_psnr:.2f} dB")


dwt_img = dwt_compress(img, threshold=30)
dwt_psnr = calculate_psnr(img, dwt_img)
print(f"DWT PSNR: {dwt_psnr:.2f} dB")


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.suptitle(
    f"Compression Techniques Comparison\n(Image Size: {img.shape})", fontsize=14
)

axes[0].imshow(img, cmap="gray")
axes[0].set_title(f"Original Image\n(Huffman Ratio: {huff_ratio:.2f}:1)")
axes[0].axis("off")

axes[1].imshow(dct_img, cmap="gray")
axes[1].set_title(f"DCT Compressed (JPEG-like)\nPSNR: {dct_psnr:.2f} dB")
axes[1].axis("off")

axes[2].imshow(dwt_img, cmap="gray")
axes[2].set_title(f"DWT Compressed (Haar)\nPSNR: {dwt_psnr:.2f} dB")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("images/output/compression_result.png")
print("Result saved as 'compression_result.png'")