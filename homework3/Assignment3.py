from cv2.gapi import RGB2HSV
import numpy as np
import torch
import torchvision
import cv2 as cv
import matplotlib.pyplot as plt


class Assignment3:
    def __init__(self) -> None:
        pass

    def torch_image_conversion(self, torch_img):
        # print("Original Image Shape:", torch_img.shape)
        # print("Original Image Data Type:", torch_img.dtype)
        img_rgb = cv.cvtColor(torch_img, cv.COLOR_BGR2RGB)
        torch_img = torch.from_numpy(img_rgb).float()
        # print("Converted Image Shape:", torch_img.shape)
        # print("Converted Image Data Type:", torch_img.dtype)

        return torch_img

    def brighten(self, torch_img):
        bright_img = torch_img + 100.0 
        bright_img = torch.clamp(bright_img, 0.0, 255.0)

        rgb_img = bright_img.numpy().astype(np.uint8)
        rgb_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)
        cv.imwrite("brightened_image.png", rgb_img)

        return bright_img

    def saturation_arithmetic(self, img):
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(rgb_img).to(torch.uint8)

        saturated_img = tensor_img + 100
        saturated_img = torch.clamp(bright_img, 0, 255)

        cv_img = saturated_img.numpy().astype(np.uint8)
        cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2BGR)
        cv.imwrite("saturated_image.png", cv_img)

        return saturated_img

    def add_noise(self, torch_img):

        return noisy_img

    def normalization_image(self, img):

        return image_norm

    def Imagenet_norm(self, img):

        return ImageNet_norm

    def dimension_rearrange(self, img):

        return rearrange

    def chain_rule(self, x, y, z):

        return df_dx, df_dy, df_dz, df_dq

    def relu(self, x, w):

        return dx, dw


if __name__ == "__main__":
    img = cv.imread("original_image.png")
    assign = Assignment3()
    torch_img = assign.torch_image_conversion(img)
    bright_img = assign.brighten(torch_img)
    saturated_img = assign.saturation_arithmetic(img)
    noisy_img = assign.add_noise(torch_img)
    image_norm = assign.normalization_image(img)
    ImageNet_norm = assign.Imagenet_norm(img)
    rearrange = assign.dimension_rearrange(img)
    df_dx, df_dy, df_dz, df_dq = assign.chain_rule(x=-2.0, y=5.0, z=-4.0)
    dx, dw = assign.relu(x=[-1.0, 2.0], w=[2.0, -3.0, -3.0])
