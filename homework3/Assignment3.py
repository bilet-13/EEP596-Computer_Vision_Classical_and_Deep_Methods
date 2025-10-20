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
        img_rgb = cv.cvtColor(torch_img, cv.COLOR_BGR2RGB)
        torch_img = torch.from_numpy(img_rgb).float()

        return torch_img

    def brighten(self, torch_img):
        bright_img = torch_img + 100.0 
        return bright_img

    def saturation_arithmetic(self, img):
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(rgb_img).to(torch.uint8)

        saturated_img = torch.clamp(tensor_img.to(torch.int16) + 100, 0, 255).to(torch.uint8)

        # write_img = saturated_img.numpy().astype(np.uint8)
        # write_img = cv.cvtColor(write_img, cv.COLOR_RGB2BGR)
        # cv.imwrite("saturated_image.png", write_img)

        return saturated_img

    def add_noise(self, torch_img):
        noise = np.random.normal(0, 100, torch_img.shape).astype(np.float32)    
        noisy_img = torch_img + torch.from_numpy(noise)
        noisy_img = torch.clamp(noisy_img, 0.0, 255.0)
        noisy_img = noisy_img / 255.0

        return noisy_img

    def normalization_image(self, img):
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = rgb_img.astype(np.float64)

        mean = img.mean(axis=(0, 1), dtype=np.float64)
        std = img.std(axis=(0, 1), dtype=np.float64)

        image_norm = (img - mean) / std

        image_norm = torch.from_numpy(image_norm).double()

        return image_norm

    def Imagenet_norm(self, img):
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)   

        torch_img = torch.from_numpy(rgb).to(torch.float64) / 255.0
        torch_img = torch_img.clamp(0.0, 1.0)

        # x = x.permute(2, 0, 1).contiguous()

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float64) 
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float64)

        ImageNet_norm = (torch_img - mean) / std 
        ImageNet_norm = ImageNet_norm.clamp(0, 1.0)
        # ImageNet_norm = ImageNet_norm.permute(1, 2, 0).contiguous()

        return ImageNet_norm

    def dimension_rearrange(self, img):
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(rgb_img).float()

        rearrange = tensor_img.permute(2, 0, 1)
        rearrange = rearrange.unsqueeze(0)

        return rearrange

    def stride(self, img):
        torch_img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  
        torch_img = torch_img.to(torch.float32)

        scharr_x = np.array([[-3, 0, 3],
                             [-10, 0, 10],
                            [-3, 0, 3]], dtype=np.float32)
        kernel = torch.from_numpy(scharr_x).flip(0, 1).unsqueeze(0).unsqueeze(0)  

        torch_image = torch.nn.functional.conv2d(torch_img, kernel, stride=2, padding=1)
        torch_image = torch_image.squeeze(0).squeeze(0)

        return torch_image
        

    # def chain_rule(self, x, y, z):

    #     return df_dx, df_dy, df_dz, df_dq

    # def relu(self, x, w):

    #     return dx, dw


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
    gray_img = cv.imread("cat_eye.jpg", cv.IMREAD_GRAYSCALE)
    stride_img = assign.stride(gray_img)
    # df_dx, df_dy, df_dz, df_dq = assign.chain_rule(x=-2.0, y=5.0, z=-4.0)
    # dx, dw = assign.relu(x=[-1.0, 2.0], w=[2.0, -3.0, -3.0])
