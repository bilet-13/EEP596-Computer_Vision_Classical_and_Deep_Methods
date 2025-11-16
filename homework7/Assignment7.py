import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import convolve2d  # TODO: use torch.nn.functional
import os

# region DEBUG
DEBUG = False


def print_debug(*args, **kwargs):
    if DEBUG:
        print(args, kwargs)


if __name__ == "__main__":
    DEBUG = os.environ.get("PYTHON_DEBUG_MODE")
    if DEBUG is not None and DEBUG.lower() == "true":
        DEBUG = True
        print("DEBUG mode is enabled")
# endregion
    if not os.path.exists("figure"):
        os.makedirs("figure")


def load_image_in_grayscale(filepath) -> torch.tensor:
    return cv.imread(filepath, cv.IMREAD_GRAYSCALE)


def sum_of_abs_diff(nparray1: np.array, nparray2: np.array) -> int:
    return (np.abs(nparray1 - nparray2)).sum().item()


def scanlines(tb_left: np.array, tb_right: np.array):
    row_idx = 152
    col_idx1 = 102
    col_len = 100
    tb_left_cropped = tb_left[row_idx][col_idx1 : col_idx1 + col_len]

    g_best = None
    d_best = None
    max_disparity = col_idx1
    for d in range(max_disparity + 1):  # TODO: check max disparity
        tb_right_cropped = tb_right[row_idx][col_idx1 - d : col_idx1 - d + col_len]
        g = sum_of_abs_diff(tb_left_cropped, tb_right_cropped)
        if g_best == None or g < g_best:
            g_best, d_best = g, d

    print(f"Best g: {g_best}, Best d: {d_best}")
    return d_best


def plot_1d_array(array, title, xlabel=None, ylabel=None, save_image=True):
    domain = range(len(array))
    plt.plot(domain, array, marker="o")
    plt.xlabel(title)
    plt.ylabel(xlabel)
    plt.title(ylabel)
    plt.grid(True)
    if save_image:
        plt.savefig(f"figure/{title}.png")
    plt.show()


def plot_2d_array_as_image(array2d: np.array, title, save_image=True):
    plt.imshow(array2d, cmap="gray")
    plt.title(title)
    plt.colorbar()
    if save_image:
        plt.savefig(f"figure/{title}.png")
    plt.show()


def shift_array(nparray: np.array, d: int) -> np.array:
    shifted = np.zeros_like(nparray)
    if d == 0:
        shifted[:, :] = nparray[:, :]
    elif d > 0:
        shifted[:, d:] = nparray[:, :-d]
    elif d < 0:
        shifted[:, : nparray.shape[1] + d] = nparray[:, -d:]
    return shifted


if DEBUG:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert (shift_array(a, 1) == [[0, 1, 2], [0, 4, 5], [0, 7, 8]]).all()
    assert (shift_array(a, 2) == [[0, 0, 1], [0, 0, 4], [0, 0, 7]]).all()


def auto_correlation(tb_right):
    max_d = 30
    auto_correlations = []
    for d in range(max_d + 1):
        abs_diff_image = np.abs(tb_right - shift_array(tb_right, d))
        auto_correlations.append(abs_diff_image[152][152])

    if DEBUG:
        plot_1d_array(auto_correlations, auto_correlation.__name__)
    return auto_correlations


# TODO
def convolve2d_torch(array: np.array, kernel_size: int):
    as_tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor(np.ones((kernel_size, kernel_size)), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    convolved = nn.functional.conv2d(as_tensor, kernel, padding=kernel_size // 2)
    if DEBUG:
        assert convolved.shape == as_tensor.shape

    return np.array(convolved.squeeze().squeeze())


def smoothing(tb_right):
    max_d = 30
    smoothed_auto_corr = []

    for d in range(max_d + 1):
        shifted = shift_array(tb_right, d)

        abs_diff_image = np.abs(tb_right - shifted)

        smoothed = convolve2d_torch(abs_diff_image, kernel_size=5)

        smoothed_auto_corr.append(smoothed[152, 152])


    return smoothed_auto_corr


def cross_correlation(tb_left, tb_right):
    max_d = 30
    cross_corr = []

    for d in range(max_d + 1):
        shifted = shift_array(tb_right, d)

        abs_diff_image = np.abs(tb_left - shifted)

        cross_corr.append(abs_diff_image[152, 152])

    return cross_corr


def disparity_map(tb_left, tb_right, plot_result=False):
    H, W = tb_left.shape
    max_d = 30

    smoothed_tensor = np.zeros((max_d + 1, H, W), dtype=np.float32)

    for d in range(max_d + 1):
        shifted = shift_array(tb_right, d)

        abs_diff = np.abs(tb_left - shifted)

        smoothed = convolve2d_torch(abs_diff, kernel_size=5)

        smoothed_tensor[d] = smoothed

    disparity = np.argmin(smoothed_tensor, axis=0)   

    disparity_img = disparity.astype(np.uint8)

    if plot_result:
        plt.imshow(disparity_img, cmap='gray')
        plt.title("Disparity Map")
        plt.colorbar()
        plt.show()

    return disparity_img


def right_left_disparity(tb_left, tb_right, plot_result=False):
    H, W = tb_left.shape
    max_d = 30

    cost_tensor = np.zeros((max_d + 1, H, W), dtype=np.float32)

    for d in range(max_d + 1):
        shifted_left = shift_array(tb_left, d)

        abs_diff = np.abs(tb_right - shifted_left)

        cost_tensor[d] = abs_diff

    disparity = np.argmin(cost_tensor, axis=0)

    disparity_img = disparity.astype(np.uint8)
    plot_2d_array_as_image(disparity_img, "Right-Left Disparity Map", save_image=True)
    
    return disparity_img



def disparity_check(tb_left, tb_right):
    dL = disparity_map(tb_left, tb_right, plot_result=False).astype(np.int32)
    dR = right_left_disparity(tb_left, tb_right, plot_result=False).astype(np.int32)

    H, W = dL.shape

    cleaned = np.zeros_like(dL, dtype=np.uint8)

    for x in range(H):
        for y in range(W):
            d = dL[x, y]

            if d == 0:
                continue

            xr = x
            yr = y - d   

            if yr < 0 or yr >= W:
                continue

            if dR[xr, yr] == d:
                cleaned[x, y] = d
            else:
                cleaned[x, y] = 0
    plot_2d_array_as_image(cleaned, "Disparity Check Result", save_image=True)

    return cleaned


def reconstruction(tb_left, tb_right):
    dL = disparity_map(tb_left, tb_right, plot_result=False).astype(np.int32)
    dR = right_left_disparity(tb_left, tb_right, plot_result=False).astype(np.int32)
    cleaned = disparity_check(tb_left, tb_right)       # already uint8
    cleaned = cleaned.astype(np.int32)

    H, W = cleaned.shape

    tb_left_color = plt.imread("tsukuba_left.png")  
    
    if tb_left_color.dtype != np.uint8:
        tb_left_color = (tb_left_color * 255).astype(np.uint8)

    mask = cleaned > 0
    ys, xs = np.where(mask)
    ds = cleaned[ys, xs].astype(np.float32)

    colors = tb_left_color[ys, xs]       # N x 3

    f = 1.0      
    B = 0.1     
    depth_scale = f * B   

    cx = W / 2.0
    cy = H / 2.0
    xy_scale = 0.002     

    X = (xs - cx) * xy_scale
    Y = -(ys - cy) * xy_scale
    Z = depth_scale / ds

    num_vertices = X.shape[0]

    with open("kermit.ply", "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for Xv, Yv, Zv, color in zip(X, Y, Z, colors):
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            f.write(f"{Xv:.6f} {Yv:.6f} {Zv:.6f} {r} {g} {b}\n")

    print(f"Saved PLY:  ({num_vertices} points)")

if __name__ == "__main__":
    tb_left = load_image_in_grayscale("tsukuba_left.png")
    tb_right = load_image_in_grayscale("tsukuba_right.png")
    scanlines(tb_left, tb_right)
    auto_correlation(tb_right)
    smoothing(tb_right)
    cross_correlation(tb_left, tb_right)
    disparity_map(tb_left, tb_right, plot_result=True)
    right_left_disparity(tb_left, tb_right, plot_result=True)
    disparity_check(tb_left, tb_right)
    reconstruction(tb_left, tb_right)
