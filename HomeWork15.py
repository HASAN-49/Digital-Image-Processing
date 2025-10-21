import numpy as np
import cv2
import matplotlib.pyplot as plt


def display_images(images, titles):
    plt.figure(figsize=(20, 20))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, len(images), i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')

        plt.subplot(2, len(images), i + 1 + len(images))
        plt.hist(img.ravel(), bins=256, range=(0, 256))
        plt.title(f"Histogram - {title}")
    plt.tight_layout()
    plt.show()



def _distance_matrix(shape):
    M, N = shape
    cy, cx = M // 2, N // 2
    y = np.arange(M) - cy
    x = np.arange(N) - cx
    X, Y = np.meshgrid(x, y)
    return np.sqrt(X*X + Y*Y)


def _max_radius(shape):
    M, N = shape
    return np.hypot(M/2.0, N/2.0)



def ideal_lpf(shape, D0):
    D = _distance_matrix(shape)
    return (D <= D0).astype(np.float32)


def ideal_hpf(shape, D0):
    return 1.0 - ideal_lpf(shape, D0)


def ideal_bpf(shape, D_low, D_high):
    return ideal_lpf(shape, D_high) * ideal_hpf(shape, D_low)



def gaussian_lpf(shape, D0):
    D = _distance_matrix(shape)
    return np.exp(-(D*D) / (2.0 * (D0**2 + 1e-12))).astype(np.float32)


def gaussian_hpf(shape, D0):
    return 1.0 - gaussian_lpf(shape, D0)


def gaussian_bpf(shape, D_low, D_high):
    return gaussian_lpf(shape, D_high) * gaussian_hpf(shape, D_low)



def butterworth_lpf(shape, D0, n):
    D = _distance_matrix(shape)
    ratio = (D / max(D0, 1e-9)) ** (2 * n)
    H = 1.0 / (1.0 + ratio)
    return H.astype(np.float32)


def butterworth_hpf(shape, D0, n):
    return 1.0 - butterworth_lpf(shape, D0, n)


def butterworth_bpf(shape, D_low, D_high, n):
    return butterworth_lpf(shape, D_high, n) * butterworth_hpf(shape, D_low, n)



def _to_float01(img_u8):
    return img_u8.astype(np.float32) / 255.0


def _from_float01(img):
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


def apply_filter_freq(img_u8, H, robust=True):
    """Apply frequency-domain filter H to grayscale uint8 image and return uint8."""
    img01 = _to_float01(img_u8)
    F = np.fft.fftshift(np.fft.fft2(img01))
    G = F * H
    g = np.real(np.fft.ifft2(np.fft.ifftshift(G)))

    if robust:
        lo, hi = np.percentile(g, [0.5, 99.5])
        if hi - lo < 1e-8:
            g = np.zeros_like(g)
        else:
            g = (g - lo) / (hi - lo)
    return _from_float01(g)



def adjust_contrast(img, mode='normal'):
    img = img.astype(np.float32)
    if mode == 'low':
        img = img * 0.5 + 64
    elif mode == 'high':
        img = (img - 128) * 2 + 128
    return np.clip(img, 0, 255).astype(np.uint8)


def sobel_energy(img_u8):
    gx = cv2.Sobel(img_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_u8, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx*gx + gy*gy))


IMAGE_PATH = 'nature1.png'
LOW_CUT_FRAC = 0.10
HIGH_CUT_FRAC = 0.30
BUTTER_N_LIST = [1, 2, 4, 8]


def run_for_variant(img_u8, label):
    M, N = img_u8.shape
    rmax = _max_radius(img_u8.shape)
    D_low = LOW_CUT_FRAC * rmax
    D_high = HIGH_CUT_FRAC * rmax

    H_ilpf = ideal_lpf((M, N), D_high)
    H_ihpf = ideal_hpf((M, N), D_low)
    H_ibpf = ideal_bpf((M, N), D_low, D_high)

    H_glpf = gaussian_lpf((M, N), D_high)
    H_ghpf = gaussian_hpf((M, N), D_low)
    H_gbpf = gaussian_bpf((M, N), D_low, D_high)

    ilpf = apply_filter_freq(img_u8, H_ilpf)
    ihpf = apply_filter_freq(img_u8, H_ihpf)
    ibpf = apply_filter_freq(img_u8, H_ibpf)

    glpf = apply_filter_freq(img_u8, H_glpf)
    ghpf = apply_filter_freq(img_u8, H_ghpf)
    gbpf = apply_filter_freq(img_u8, H_gbpf)

    lpf_imgs = [img_u8, ilpf, glpf]
    lpf_titles = [f"{label} (orig)", "Ideal LPF", "Gaussian LPF"]

    hpf_imgs = [img_u8, ihpf, ghpf]
    hpf_titles = [f"{label} (orig)", "Ideal HPF", "Gaussian HPF"]

    bpf_imgs = [img_u8, ibpf, gbpf]
    bpf_titles = [f"{label} (orig)", "Ideal BPF", "Gaussian BPF"]

    for n in BUTTER_N_LIST:
        blpf = apply_filter_freq(img_u8, butterworth_lpf((M, N), D_high, n))
        bhpf = apply_filter_freq(img_u8, butterworth_hpf((M, N), D_low, n))
        bbpf = apply_filter_freq(img_u8, butterworth_bpf((M, N), D_low, D_high, n))

        lpf_imgs.append(blpf);
        lpf_titles.append(f"Butter LPF (n={n})")

        hpf_imgs.append(bhpf);
        hpf_titles.append(f"Butter HPF (n={n})")

        bpf_imgs.append(bbpf);
        bpf_titles.append(f"Butter BPF (n={n})")

    print(f"\n=== {label} – LPF comparison ===")
    for t, im in zip(lpf_titles[1:], lpf_imgs[1:]):
        print(f"  {t:22s}  SobelEnergy={sobel_energy(im):.6f}")
    display_images(lpf_imgs, lpf_titles)

    print(f"\n=== {label} – HPF comparison ===")
    for t, im in zip(hpf_titles[1:], hpf_imgs[1:]):
        print(f"  {t:22s}  SobelEnergy={sobel_energy(im):.6f}")
    display_images(hpf_imgs, hpf_titles)

    print(f"\n=== {label} – BPF comparison ===")
    for t, im in zip(bpf_titles[1:], bpf_imgs[1:]):
        print(f"  {t:22s}  SobelEnergy={sobel_energy(im):.6f}")
    display_images(bpf_imgs, bpf_titles)


def main():
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{IMAGE_PATH}'")

    low = adjust_contrast(img, 'low')
    norm = adjust_contrast(img, 'normal')
    high = adjust_contrast(img, 'high')

    run_for_variant(low,  'Low Contrast')
    run_for_variant(norm, 'Normal Contrast')
    run_for_variant(high, 'High Contrast')


if __name__ == "__main__":
    main()
