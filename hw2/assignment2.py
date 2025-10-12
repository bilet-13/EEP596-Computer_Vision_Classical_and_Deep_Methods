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

  def gussian_operation(self, img):
    h, w = img.shape
    kernel = [0.25, 0.5, 0.25]

    grid = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            left = img[y, x - 1] if x - 1 >= 0 else 0
            mid = img[y, x]
            right = img[y, x + 1] if x + 1 < w else 0

            grid[y, x] = np.dot([left, mid, right], kernel)

    result = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            up = grid[y - 1, x] if y - 1 >= 0 else 0
            mid = grid[y, x]
            down = grid[y + 1, x] if y + 1 < h else 0

            val = np.dot([up, mid, down], kernel) 
            val = int(round(val))

            result[y, x] = np.clip(val, 0, 255)

    return result 

  def gaussian_blur(self):
    """
    Apply Gaussian blur to the image iteratively.
    """
    # kernel = # 1D Gaussian kernel
    image = self.cat_eye
        
    self.blurred_images = []
    for i in range(5):
        # Apply convolution
        image = self.gussian_operation(image)
        
        # Store the blurred image
        self.blurred_images.append(image)
        
        cv2.imwrite(f'gaussain blur {i}.jpg', image)
    return self.blurred_images

  #def gaussian_derivative_vertical(self):
  #  # Define kernels
    
  #  # Store images
  #  self.vDerive_images = []
  #  for i in range(5):
  #    # Apply horizontal and vertical convolution
  #    # image =
      
  #    # self.vDerive_images.append(image)
  #    #cv2.imwrite(f'vertical {i}.jpg', image)
  #  return self.vDerive_images

  #def gaussian_derivative_horizontal(self):
  #  #Define kernels

  #  # Store images after computing horizontal derivative
  #  self.hDerive_images = []

  #  for i in range(5):

  #    # Apply horizontal and vertical convolution
  #    # image =

  #    self.hDerive_images.append(image)
  #    #cv2.imwrite(f'horizontal {i}.jpg', image)
  #  return self.hDerive_images

  #def gradient_magnitute(self):
  #  # Store the computed gradient magnitute
  #  self.gdMagnitute_images =[]
  #  for i, (vimg, himg) in enumerate(zip(self.vDerive_images, self.hDerive_images)):
  #    image = 
  #    self.gdMagnitute_images.append(image)
  #    #cv2.imwrite(f'gradient {i}.jpg', image)
  #  return self.gdMagnitute_images
    
  #def scipy_convolve(self):
  #  # Define the 2D smoothing kernel
   
  #  # Store outputs
  #  self.scipy_smooth = []

  #  for i in range(5):
  #    # Perform convolution
  #    image=
  #    self.scipy_smooth.append(image)
  #    #cv2.imwrite(f'scipy smooth {i}.jpg', image)
  #  return self.scipy_smooth

  #def box_filter(self, num_repetitions):
  #  # Define box filter
  #  box_filter = [1, 1, 1]
  #  out = [1, 1, 1]

  #  for _ in range(num_repetitions):
  #    # Perform 1D conlve
  #    out =

  #  return out

if __name__ == "__main__":
    ass = ComputerVisionAssignment()
    # # Task 1 floodfill
    floodfill_img = ass.floodfill((168, 155))

    # Task 2 Convolution for Gaussian smoothing.
    blurred_imgs = ass.gaussian_blur()

    # # Task 3 Convolution for differentiation along the vertical direction
    # # vertical_derivative = ass.gaussian_derivative_vertical()

    # # Task 4 Differentiation along another direction along the horizontal direction
    # horizontal_derivative = ass.gaussian_derivative_horizontal()

    # # Task 5 Gradient magnitude.
    # Gradient_magnitude = ass.gradient_magnitute()

    # # Task 6 Built-in convolution
    # scipy_convolve = ass.scipy_convolve()

    # # Task 7 Repeated box filtering
    # box_filter = ass.box_filter(5)
