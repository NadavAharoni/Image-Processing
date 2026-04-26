import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import correlate2d

def initialize_kernel():
    return np.array([
        [-1,  2,  1],
        [-2,  1, -3],
        [ 3,  0, -1]
    ], dtype=np.float32)

def get_image():
    return np.array([
        [103, 102, 101, 100],
        [104, 103, 102, 101],
        [ 53,  52,  51,  50],
        [ 45,  53,  52,  51]
    ], dtype=np.uint8)

def cross_correlate_loop(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = h - kh + 1
    out_w = w - kw + 1
    result = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            s = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    s += float(image[i+ki, j+kj]) * kernel[ki, kj]
            result[i, j] = s
    return result

def cross_correlate_loop_with_np(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = h - kh + 1
    out_w = w - kw + 1
    result = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i+kh, j:j+kw]
            result[i, j] = np.sum(patch * kernel)
    return result

def cross_correlate_np(image, kernel):
    windows = sliding_window_view(image, kernel.shape)  # shape: (2, 2, 3, 3)
    result = np.sum(windows * kernel, axis=(2, 3))
    return result.astype(np.float32)

def cross_correlate_scipy(image, kernel):
    result = correlate2d(image.astype(np.float32), kernel, mode='valid')
    return result.astype(np.float32)

def compare_cross_correlations(image, kernel):
    r_loop  = cross_correlate_loop(image, kernel)
    r_np    = cross_correlate_np(image, kernel)
    r_scipy = cross_correlate_scipy(image, kernel)
    ok_1 = np.allclose(r_loop, r_np)
    ok_2 = np.allclose(r_loop, r_scipy)
    return r_loop, r_np, r_scipy, bool(ok_1 and ok_2)

if __name__ == "__main__":
    kernel = initialize_kernel()
    image  = get_image()
    r_loop, r_np, r_scipy, all_match = compare_cross_correlations(image, kernel)
    print("Loop result:\n",  r_loop)
    print("NumPy result:\n", r_np)
    print("SciPy result:\n", r_scipy)
    print(f"\nLoop == NumPy : {np.allclose(r_loop, r_np)}")
    print(f"Loop == SciPy : {np.allclose(r_loop, r_scipy)}")
    print("All match!" if all_match else "⚠️  MISMATCH!")

