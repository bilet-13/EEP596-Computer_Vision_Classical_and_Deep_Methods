import cv2
import numpy as np
import os


class ComputerVisionAssignment:
    def __init__(self, image_path, binary_image_path):
        self.image = cv2.imread(image_path)
        self.binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    def check_package_versions(self):
        # Ungraded
        import numpy as np
        import matplotlib
        import cv2

        # print(np.__version__)
        # print(matplotlib.__version__)
        # print(cv2.__version__)

    def load_and_analyze_image(self):
        """
        Fill your code here

        """
        Image_data_type = self.image.dtype
        Pixel_data_type = self.image[0, 0].dtype
        Image_shape = self.image.shape

        print(f"Image data type: {Image_data_type}")
        print(f"Pixel data type: {Pixel_data_type}")
        print(f"Image dimensions: {Image_shape}")

        # return Image_data_type, Pixel_data_type, Image_shape

    def create_red_image(self):
        """
        Fill your code here

        """
        red_image = self.image.copy()
        red_image[:, :, 1] = 0
        red_image[:, :, 0] = 0

        return red_image

    def create_photographic_negative(self):
        """
        Fill your code here

        """
        negative_image = 255 - self.image
        return negative_image

    def swap_color_channels(self):
        """
        Fill your code here

        """
        swapped_image = self.image.copy()
        swapped_image[:, :,0], swapped_image[:, :,2] = self.image[:, :,2], self.image[:, :,0]
        return swapped_image

    def foliage_detection(self):
        """
        Fill your code here

        """
        return foliage_image

    def shift_image(self):
        """
        Fill your code here

        """
        return shifted_image

    def rotate_image(self):
        """
        Fill your code here

        """
        return rotated_image

    def similarity_transform(self, scale, theta, shift):
        """
        Fill your code here

        """
        return transformed_image

    def convert_to_grayscale(self):
        """
        Fill your code here

        """
        return gray_image

    def compute_moments(self):
        """
        Fill your code here

        """
        # Print the results
        # print("First-Order Moments:")
        # print(f"Standard (Raw) Moments: M00 = {m00}, M10 = {m10}, M01 = {m01}")
        # print("Centralized Moments:")
        # print(f"x_bar = {x_bar}, y_bar = {y_bar}")
        # print("Second-Order Centralized Moments:")
        # print(f"mu20 = {mu20}, mu02 = {mu02}, mu11 = {mu11}")

        return m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11

    def compute_orientation_and_eccentricity(self):
        """
        Fill your code here

        """

        return orientation, eccentricity, glasses_with_ellipse


if __name__ == "__main__":

    assignment = ComputerVisionAssignment("original_image.png", "binary_image.png")

    # Task 0: Check package versions
    assignment.check_package_versions()

    # Task 1: Load and analyze the image
    assignment.load_and_analyze_image()

    # Task 2: Create a red image
    red_image = assignment.create_red_image()
    # cv2.imwrite("red_image.png", red_image)

    # Task 3: Create a photographic negative
    negative_image = assignment.create_photographic_negative()
    # cv2.imwrite("negative_image.png", negative_image)

    # Task 4: Swap color channels
    swapped_image = assignment.swap_color_channels()
    cv2.imwrite("swapped_image.png", swapped_image)

    # Task 5: Foliage detection
    foliage_image = assignment.foliage_detection()

    # Task 6: Shift the image
    shifted_image = assignment.shift_image()

    # Task 7: Rotate the image
    rotated_image = assignment.rotate_image()

    # Task 8: Similarity transform
    transformed_image = assignment.similarity_transform(
        scale=2.0, theta=45.0, shift=[100, 100]
    )

    # Task 9: Grayscale conversion
    gray_image = assignment.convert_to_grayscale()

    glasses_assignment = ComputerVisionAssignment(
        "glasses_outline.png", "glasses_outline.png"
    )

    # Task 10: Moments of a binary image
    glasses_assignment.compute_moments()

    # Task 11: Orientation and eccentricity of a binary image
    orientation, eccentricity, glasses_with_ellipse = (
        glasses_assignment.compute_orientation_and_eccentricity()
    )
