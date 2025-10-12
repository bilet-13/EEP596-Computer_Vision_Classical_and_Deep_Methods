# -*- coding: utf-8 -*-
"""EEP 596 HW2
"""

import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt

class ComputerVisionAssignment():
  def __init__(self) -> None:
    self.ant_img = cv2.imread('ant_outline.png')
    self.cat_eye = cv2.imread('cat_eye.jpg', cv2.IMREAD_GRAYSCALE)

  def floodfill(self, seed = (0, 0)):
    fill_color = (0, 0, 255)   # (B, G, R)

    gray_img = cv2.cvtColor(self.ant_img, cv2.COLOR_BGR2GRAY)
    output_image = self.ant_img.copy()

    h, w = gray_img.shape 
    stack = [seed]

    direction = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while stack:
        x, y = stack.pop()

        output_image[y][x] = fill_color

        for dx, dy in direction:
            nx = x + dx
            ny = y + dy

            if nx >= 0 and nx < w and ny >= 0 and ny < h and gray_img[ny][nx] == 255:
                gray_img[ny][nx] = 0
                stack.append((nx, ny))

    cv2.imwrite('floodfille.jpg', output_image)
    return output_image

  def gussian_operation(self, img, kernel_x, kernel_y):
    h, w = img.shape

    grid = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            left = img[y, x - 1] if x - 1 >= 0 else 0
            mid = img[y, x]
            right = img[y, x + 1] if x + 1 < w else 0

            grid[y, x] = np.dot([left, mid, right], kernel_x)

    result = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            up = grid[y - 1, x] if y - 1 >= 0 else 0
            mid = grid[y, x]
            down = grid[y + 1, x] if y + 1 < h else 0

            val = np.dot([up, mid, down], kernel_y) 

            result[y, x] = val

    return result 

  def gaussian_blur(self):
    """
    Apply Gaussian blur to the image iteratively.
    """
    # kernel = # 1D Gaussian kernel
    image = self.cat_eye
        
    self.blurred_images = []
    gussian_kernel = [0.25, 0.5, 0.25]
    for i in range(5):
        # Apply convolution
        image = self.gussian_operation(image, gussian_kernel, gussian_kernel)
        
        # Store the blurred image
        self.blurred_images.append(image)

        image = np.rint(image)
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        cv2.imwrite(f'gaussain blur {i}.png', image)
    return self.blurred_images

  def gaussian_derivative_vertical(self):
    # Define kernels
    gussian_kernel = [0.25, 0.5, 0.25]
    soblel_kernel = [0.5, 0, -0.5] # flipped sober kernel for vertical derivative
    # Store images
    self.vDerive_images = []
    for i in range(5):
      # Apply horizontal and vertical convolution
      image = cv2.imread(f'gaussain blur {i}.png', cv2.IMREAD_GRAYSCALE)

      image = self.gussian_operation(image, gussian_kernel, soblel_kernel)

      image = 2 * image + 127
      image = np.rint(image)
      image = np.clip(image, 0, 255).astype(np.uint8)
      
      self.vDerive_images.append(image)


      cv2.imwrite(f'vertical {i}.png', image)

    return self.vDerive_images

  def gaussian_derivative_horizontal(self):
    #Define kernels
    gussian_kernel = [0.25, 0.5, 0.25]
    soblel_kernel = [0.5, 0, -0.5] # flipped sober kernel for horizontal derivative

    # Store images after computing horizontal derivative
    self.hDerive_images = []

    for i in range(5):
      # Apply horizontal and vertical convolution
      image = cv2.imread(f'gaussain blur {i}.png', cv2.IMREAD_GRAYSCALE)

      image = self.gussian_operation(image, soblel_kernel, gussian_kernel)
      
      image = 2 * image + 127
      image = np.rint(image)
      image = np.clip(image, 0, 255).astype(np.uint8)

      self.hDerive_images.append(image)

      cv2.imwrite(f'horizontal {i}.png', image)

    return self.hDerive_images

  def gradient_magnitute(self):
    # Store the computed gradient magnitute
    self.gdMagnitute_images =[]

    gussian_kernel = [0.25, 0.5, 0.25]
    soblel_kernel = [0.5, 0, -0.5] # flipped sober kernel for vertical derivative

    for i in range(5):
      blur_image = cv2.imread(f'gaussain blur {i}.png', cv2.IMREAD_GRAYSCALE)

      v_image = self.gussian_operation(blur_image, gussian_kernel, soblel_kernel)
      h_image = self.gussian_operation(blur_image, soblel_kernel, gussian_kernel)

      image = np.zeros(v_image.shape, dtype=np.float32)
      h, w = v_image.shape

      for y in range(h):
        for x in range(w):
          image[y, x] = abs(v_image[y, x]) + abs(h_image[y, x])

      image = 4 * image 
      image = np.rint(image)
      image = np.clip(image, 0, 255).astype(np.uint8)

      self.gdMagnitute_images.append(image)

      cv2.imwrite(f'gradient {i}.jpg', image)

    return self.gdMagnitute_images
    
  def scipy_convolve(self):
    # Define the 2D smoothing kernel
    gussian_kernel = [0.25, 0.5, 0.25]
    sober_kernel = [0.5, 0, -0.5]

    kernel_2d = np.outer(sober_kernel, gussian_kernel)

    # Store outputs
    self.scipy_smooth = []

    for i in range(5):
      # Perform convolution
      image = cv2.imread(f'gaussain blur {i}.png', cv2.IMREAD_GRAYSCALE)

      image = scipy.signal.convolve2d(image,kernel_2d, mode='same', boundary='fill', fillvalue=0)

      image = 2 * image + 127
      image = np.rint(image)
      image = np.clip(image, 0, 255).astype(np.uint8)

      self.scipy_smooth.append(image)

      cv2.imwrite(f'scipy smooth {i}.jpg', image)

    return self.scipy_smooth

  def box_filter(self, num_repetitions):
    # Define box filter
    box_filter = [1, 1, 1]
    out = [1, 1, 1]

    for _ in range(num_repetitions):
      # Perform 1D conlve
      convolution_result = np.zeros(len(out) + len(box_filter) - 1, dtype=np.float32)

      for j in range(len(convolution_result)):
        for k in range(len(box_filter)):
          idx = j + k + 1 - len(box_filter)

          if idx >= 0 and idx < len(out):
            convolution_result[j] += out[idx] * box_filter[k] 

      out = convolution_result

    return out

if __name__ == "__main__":
    ass = ComputerVisionAssignment()
    # # Task 1 floodfill
    floodfill_img = ass.floodfill((168, 155))

    # Task 2 Convolution for Gaussian smoothing.
    blurred_imgs = ass.gaussian_blur()

    # # Task 3 Convolution for differentiation along the vertical direction
    vertical_derivative = ass.gaussian_derivative_vertical()

    # # Task 4 Differentiation along another direction along the horizontal direction
    horizontal_derivative = ass.gaussian_derivative_horizontal()

    # # Task 5 Gradient magnitude.
    Gradient_magnitude = ass.gradient_magnitute()

    # # Task 6 Built-in convolution
    scipy_convolve = ass.scipy_convolve()

    # # Task 7 Repeated box filtering
    box_filter = ass.box_filter(5)
