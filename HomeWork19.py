import cv2
import numpy as np
import os
import scipy.fftpack
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import heapq
from collections import defaultdict

IMAGE_PATH = "/home/hasan/Desktop/Hasan/7th Semester/CSE4161_2_DIP/Code/original.png"

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


def resize_image(image, width=256):
    h, w = image.shape[:2]
    aspect_ratio = h / w
    new_h = int(width * aspect_ratio)
    return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)


def calculate_psnr(original, compressed):
    mse_val = np.mean((original - compressed) ** 2)
    if mse_val == 0:
        return float("inf")
    
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_val))
    
    return psnr_value


def calculate_metrics(original, compressed):
    if original.shape != compressed.shape:
        h, w, _ = original.shape
        compressed = cv2.resize(compressed, (w, h))

    orig_f = original.astype(np.float64)
    comp_f = compressed.astype(np.float64)

    mse_val = mse(orig_f, comp_f)

    if mse_val == 0:
        psnr_val = float("inf")
    else:
        psnr_val = psnr(orig_f, comp_f, data_range=255)

    ssim_val = ssim(orig_f, comp_f, data_range=255, channel_axis=2)

    return mse_val, psnr_val, ssim_val


def compress_rle(image):
    return image.copy()


def compress_dct(image, quantization_factor=20):
    h, w, c = image.shape
    compressed = np.zeros_like(image, dtype=np.float64)

    Q = np.array(
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
    ) * (quantization_factor / 10.0)

    for k in range(c):
        h_pad = (8 - h % 8) % 8
        w_pad = (8 - w % 8) % 8
        channel = np.pad(image[:, :, k], ((0, h_pad), (0, w_pad)), "edge").astype(float)

        restored_channel = np.zeros_like(channel)

        for i in range(0, channel.shape[0], 8):
            for j in range(0, channel.shape[1], 8):
                block = channel[i : i + 8, j : j + 8] - 128
                dct_block = scipy.fftpack.dct(
                    scipy.fftpack.dct(block.T, norm="ortho").T, norm="ortho"
                )
                dct_quant = np.round(dct_block / Q)
                dct_dequant = dct_quant * Q
                idct_block = scipy.fftpack.idct(
                    scipy.fftpack.idct(dct_dequant.T, norm="ortho").T, norm="ortho"
                )
                restored_channel[i : i + 8, j : j + 8] = idct_block + 128

        compressed[:, :, k] = restored_channel[:h, :w]

    return np.clip(compressed, 0, 255).astype(np.uint8)


def compress_dwt(image, threshold=30):
    compressed = np.zeros_like(image)
    for k in range(3):
        coeffs = pywt.wavedec2(image[:, :, k], "haar", level=2)

        coeffs_thresh = list(coeffs)
        for i in range(1, len(coeffs)):
            coeffs_thresh[i] = tuple(
                map(lambda x: pywt.threshold(x, threshold, mode="soft"), coeffs[i])
            )
        rec = pywt.waverec2(coeffs_thresh, "haar")

        h, w, _ = image.shape
        compressed[:, :, k] = np.clip(rec[:h, :w], 0, 255)

    return compressed.astype(np.uint8)


img = cv2.imread(IMAGE_PATH)
if img is None:
    print("Error: Image not found. Please check the file path.")
else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = resize_image(img, width=256)

    huff_ratio, huff_bits = huffman_compression_stats(img_resized)
    print(f"Huffman Compression Ratio: {huff_ratio:.2f}")

    dct_img = compress_dct(img_resized, quantization_factor=20)
    dct_psnr = calculate_psnr(img_resized, dct_img)
    print(f"DCT PSNR: {dct_psnr:.2f} dB")

    dwt_img = compress_dwt(img_resized, threshold=30)
    dwt_psnr = calculate_psnr(img_resized, dwt_img)
    print(f"DWT PSNR: {dwt_psnr:.2f} dB")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.suptitle(
        f"Compression Techniques Comparison\n(Image Size: {img_resized.shape})", fontsize=14
    )

    axes[0].imshow(img_resized)
    axes[0].set_title(f"Original Image\n(Huffman Ratio: {huff_ratio:.2f}:1)")
    axes[0].axis("off")

    axes[1].imshow(dct_img)
    axes[1].set_title(f"DCT Compressed (JPEG-like)\nPSNR: {dct_psnr:.2f} dB")
    axes[1].axis("off")

    axes[2].imshow(dwt_img)
    axes[2].set_title(f"DWT Compressed (Haar)\nPSNR: {dwt_psnr:.2f} dB")
    axes[2].axis("off")

    plt.tight_layout()
    output_path = "images/output/compression_result.png"
    plt.savefig(output_path)
    print(f"Result saved as '{output_path}'")
