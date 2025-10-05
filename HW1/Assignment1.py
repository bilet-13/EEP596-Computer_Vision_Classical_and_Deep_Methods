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
        foliage_image = self.image.copy()
        mask = (self.image[:, :, 1] >= 50) & \
               (self.image[:, :, 0] < 50) & \
               (self.image[:, :, 2] < 50)

        foliage_image[:, :, :] = np.where(mask[:, :, None], 255, 0)

        return foliage_image

    def shift_image(self):
        """
        Fill your code here

        """
        shifted_image = np.zeros_like(self.image)
        shifted_image[100:, 200:, :] = self.image[:-100, :-200, :]

        return shifted_image

    def rotate_image(self):
        """
        Fill your code here

        """
        rotated_image = self.image.copy()

        rotated_image = np.transpose(rotated_image, (1, 0, 2))
        rotated_image = np.flip(rotated_image, axis=1)

        return rotated_image

    def similarity_transform(self, scale, theta, shift):
        """
        Fill your code here

        """
        height, width = self.image.shape[:2]
        center = (width / 2.0, height / 2.0)

        transformMatrix = cv2.getRotationMatrix2D(center, -theta, scale)
        transformMatrix[0, 2] += shift[0]
        transformMatrix[1, 2] += shift[1]
        
        transformed_image = cv2.warpAffine(
            self.image, 
            transformMatrix, 
            (width, height),
            flags=cv2.INTER_NEAREST,
        ) 

        return transformed_image

    def convert_to_grayscale(self):
        """
        Fill your code here

        """
        weights = np.array([1.0, 6.0, 3.0])

        gray_image = np.dot(self.image.astype(np.float32), weights) / 10

        gray_image = np.clip(gray_image, 0, 255)

        return gray_image.astype(np.uint8)

    def compute_moments(self):
        """
        Fill your code here

        """
        mask = self.binary_image > 0
        height, width = self.binary_image.shape[:2]
        y_idxs, x_idxs = np.indices((height, width))

        m00 = np.sum(mask)
        m10 = np.sum(x_idxs * mask)
        m01 = np.sum(y_idxs * mask)
        
        x_bar = m10 / m00
        y_bar = m01 / m00

        mu20 = np.sum(((x_idxs - x_bar) ** 2) * mask)
        mu02 = np.sum(((y_idxs - y_bar) ** 2) * mask)
        mu11 = np.sum((x_idxs - x_bar) * (y_idxs - y_bar) * mask)
        # Print the results
        print("First-Order Moments:")
        print(f"Standard (Raw) Moments: M00 = {m00}, M10 = {m10}, M01 = {m01}")
        print("Centralized Moments:")
        print(f"x_bar = {x_bar}, y_bar = {y_bar}")
        print("Second-Order Centralized Moments:")
        print(f"mu20 = {mu20}, mu02 = {mu02}, mu11 = {mu11}")

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
    cv2.imwrite("foliage_image.png", foliage_image)

    # Task 6: Shift the image
    shifted_image = assignment.shift_image()
    cv2.imwrite("shifted_image.png", shifted_image) 

    # Task 7: Rotate the image
    rotated_image = assignment.rotate_image()
    cv2.imwrite("rotated_image.png", rotated_image)

    # Task 8: Similarity transform
    transformed_image = assignment.similarity_transform(
        scale=2.0, theta=45.0, shift=[100, 100]
    )
    cv2.imwrite("transformed_image.png", transformed_image)

    # Task 9: Grayscale conversion
    gray_image = assignment.convert_to_grayscale()
    cv2.imwrite("gray_image.png", gray_image)

    glasses_assignment = ComputerVisionAssignment(
        "glasses_outline.png", "glasses_outline.png"
    )

    # Task 10: Moments of a binary image
    glasses_assignment.compute_moments()

    # Task 11: Orientation and eccentricity of a binary image
    orientation, eccentricity, glasses_with_ellipse = (
        glasses_assignment.compute_orientation_and_eccentricity()
    )
