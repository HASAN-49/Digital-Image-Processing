import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

def dft(signal):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, signal)

def fft(signal):
    N = len(signal)
    if N <= 1:
        return signal
    elif N % 2 != 0:
        return dft(signal)
    
    even = fft(signal[::2])
    odd = fft(signal[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    
    return np.concatenate([
        even + factor[:N//2] * odd,
        even - factor[:N//2] * odd
    ])

def dft2(image):
    M, N = image.shape
    dft_rows = np.array([dft(row) for row in image])
    dft_cols = np.array([dft(col) for col in dft_rows.T]).T
    return dft_cols

def fft2(image):
    M, N = image.shape
    fft_rows = np.array([fft(row) for row in image])
    fft_cols = np.array([fft(col) for col in fft_rows.T]).T
    return fft_cols

img = cv2.imread('nature1.png', cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (64, 64))
img = img.astype(float)

print("Processing... Please wait.")

start = time.time()
dft_result = dft2(img)
end = time.time()
print(f"DFT computation time: {end - start:.4f} seconds")

start = time.time()
fft_result = fft2(img)
end = time.time()
print(f"FFT computation time: {end - start:.4f} seconds")

dft_shift = np.fft.fftshift(dft_result)
fft_shift = np.fft.fftshift(fft_result)

magnitude_dft = np.log(1 + np.abs(dft_shift))
magnitude_fft = np.log(1 + np.abs(fft_shift))

plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(magnitude_dft, cmap='gray')
plt.title('DFT Spectrum (from scratch)')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(magnitude_fft, cmap='gray')
plt.title('FFT Spectrum (from scratch)')
plt.axis('off')

plt.tight_layout()
plt.show()

np_fft = np.fft.fft2(img)
equal = np.allclose(fft_result, np_fft, atol=1e-6)
print("FFT result matches NumPy FFT:", equal)
