import glob
import os
import pandas as pd
import cv2
import numpy as np
import math
import scipy.ndimage
import matplotlib.pyplot as plt
import scipy.ndimage
import matplotlib.pyplot as plt
import tensorflow as tf

class CrackDetector:
    def _init_(self, image_path, fudge_factor=5, sigma=200):
        self.image = self.load_and_preprocess_image(image_path)
        self.fudge_factor = fudge_factor
        self.sigma = sigma
        self.kernel_size = 2 * math.ceil(2 * self.sigma) + 1

    def load_and_preprocess_image(self, image_path, target_size=(2048, 2048)):
        """Load and preprocess the image using TensorFlow operations."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.resize(image, target_size)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return self.apply_preprocessing(image)

    def apply_preprocessing(self, image):
        """Convert TensorFlow tensor to a float32 NumPy array and normalize."""
        image_np = image.numpy()  # Convert TF tensor to NumPy array
        image_np = image_np / 255.0  # Normalize to range [0, 1] if not already
        return image_np

    def read_and_prepare_image(self):
        """Prepare the image for processing."""
        # Ensure the image is in the correct format and type for OpenCV operations
        gray_image = (self.image * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
        gray_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        gray_image = cv2.bilateralFilter(gray_image, 5, 75, 75)
        return gray_image

    # Additional methods would continue here...

    def rotate_image(self,image, angle):
        """
        Rotates an image (clockwise) by a given angle around its center.

        :param image: The image to rotate (numpy array).
        :param angle: The angle in degrees. Positive values mean counter-clockwise rotation.
        :return: The rotated image.
        """
        # Get the image dimensions (the last two dimensions)
        height, width = image.shape[:2]

        # Point to center the image
        center = (width // 2, height // 2)

        # Rotation matrix using OpenCV
        # cv2.getRotationMatrix2D needs center, angle, scale
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # New dimensions of the image
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])

        # Compute the new bounding dimensions of the image
        nWidth = int((height * sin) + (width * cos))
        nHeight = int((height * cos) + (width * sin))

        # Adjust the rotation matrix to take into account translation
        matrix[0, 2] += (nWidth / 2) - center[0]
        matrix[1, 2] += (nHeight / 2) - center[1]

        # Perform the affine transformation (rotate the image)
        rotated = cv2.warpAffine(image, matrix, (nWidth, nHeight))
        return rotated
    def read_and_prepare_image(self):
        gray_image = self.image
        if gray_image is None:
            raise ValueError("Error loading the image.")

        gray_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        gray_image = np.uint8(gray_image)
        gray_image = cv2.bilateralFilter(gray_image, 5, 75, 75)
        return gray_image

    def apply_gaussian_blur(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def detect_edges(self, image):
        blurred_image = self.apply_gaussian_blur(image)
        edge_enhanced = cv2.subtract(image, blurred_image)
        sobelx = cv2.Sobel(edge_enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(edge_enhanced, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(sobelx, sobely)
        mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        mag=cv2.convertScaleAbs(mag)
        ang = np.arctan2(sobely, sobelx)
        mag = self.orientated_non_max_suppression(mag, ang)
        return mag


    def orientated_non_max_suppression(self, mag, ang):
        ang_quant = np.round(ang / (np.pi / 4)) % 4
        windows = {
            0: np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
            1: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            2: np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),
            3: np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        }
        for key, win in windows.items():
            mag[ang_quant == key] = self.non_max_suppression(mag, win)[ang_quant == key]
        return mag

    @staticmethod
    def non_max_suppression(data, win):
        data_max = scipy.ndimage.maximum_filter(data, footprint=win, mode='constant')
        data_max[data != data_max] = 0
        return data_max



    def threshold_and_close(self, mag):
        threshold = 4 * self.fudge_factor * np.mean(mag)
        mag[mag < threshold] = 0
        kernel = np.ones((5, 5), np.uint8)

        return cv2.morphologyEx(mag.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    def find_contours(self, processed_image, min_area=500, max_aspect_ratio=4, max_circularity=4, roi=None):
        """
        Finds and returns contours in the processed image that are larger than a specified minimum area,
        are not predominantly vertical based on an aspect ratio threshold, and have a circularity below a given threshold.

        :param processed_image: The image from which contours are to be extracted.
        :param min_area: The minimum area a contour must have to be included in the result.
        :param max_aspect_ratio: The maximum aspect ratio (width/height) for a contour to be considered non-vertical.
        :param max_circularity: The maximum circularity for a contour to be considered (helps ignore round objects).
        :param roi: A tuple (x, y, width, height) defining the region of interest where cracks are expected.
        :return: A list of contours that meet the area, aspect ratio, and circularity criteria.
        """
        # Find all contours in the image
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If a region of interest (ROI) is provided, crop the processed image to that ROI
        if roi is not None:
            x_roi, y_roi, w_roi, h_roi = roi
            contours = [contour for contour in contours if cv2.boundingRect(contour) == (x_roi, y_roi, w_roi, h_roi)]

        # Filter contours based on area, aspect ratio, and circularity
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if aspect_ratio > max_aspect_ratio:  # Ensure the contour is not too vertical
                    # Calculate the perimeter and circularity
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter != 0 else 0
                    if circularity < max_circularity:  # Filter out very circular (round) contours
                        filtered_contours.append(contour)

        return filtered_contours

    def detect_cracks(self):

        gray_image = self.read_and_prepare_image()
        mag = self.detect_edges(gray_image)
        processed_image = self.threshold_and_close(mag)
        contours = self.find_contours(processed_image)
        return contours

    def visualize_ellipse(self,image_path, contour, ellipse):
        image = self.image
        # Draw the contour
        cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)  # Blue
        # Draw the fitted ellipse
        cv2.ellipse(image, ellipse, (0, 255, 0), 2)  # Green

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Ellipse Angle: {ellipse[2]}')
        plt.show()

    def visualize_longest_contour1(self, image, longest_contour, image_number, angle):

        """Draw the longest contour on the image and visualize it with start/end markers."""

        contour_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_image, [longest_contour], -1, (0, 255, 0), 2)  # Draw longest contour in green

        x, y, w, h = cv2.boundingRect(longest_contour)
        start_x = x
        end_x = x + w
        longest_length = self.pixels_to_centimeters(end_x, 0, 2000, 34, 100) - 10
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
        plt.title(f'Original Image #{image_number}')
        plt.axis('on')  # Optionally display the axis



        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Image #{image_number}: Detect Crack (Length: {longest_length:.3f} cm, Angle={angle:.2f}°)')
        plt.axvline(x=start_x, color='red', linestyle='--', linewidth=2)
        plt.axvline(x=end_x, color='red', linestyle='--', linewidth=2)
        plt.axis('on')  # Optionally display the axis



        plt.show()

    def angle_from_min_area_rect_reactangle(self, contour):
        rect = cv2.minAreaRect(contour)
        angle = rect[2]  # Angle to the horizontal

        # If the width is greater than the height, we adjust the angle.
        # This is because cv2.minAreaRect could return a rectangle where the width is considered
        # to be the smaller of the two sides (hence a negative angle up to -90 degrees).
        # We adjust the angle to make it refer to the longer side of the rectangle.
        if rect[1][0] > rect[1][1]:  # if width > height
            angle = 90 + angle  # normalize angle to be from the longer side

        return angle
    def angle_from_min_area_rect(self, contour):
        if len(contour) < 5:
            print("Not enough points to fit an ellipse.")
            return self.angle_from_min_area_rect_reactangle(contour)

        # Fit an ellipse to the contour
        ellipse = cv2.fitEllipse(contour)
        if ellipse[2]==0 :
            return self.angle_from_min_area_rect_reactangle(contour)

        # self.visualize_ellipse(self.image_path, contour,ellipse)



        return ellipse[2]

    def region_of_interest1(self,start_row, end_row, start_col, end_col):
        """Extracts a region of interest from the loaded image."""
        return self.image[start_row:end_row, start_col:end_col]

    def find_crack_end_width(self,contour, orientation='horizontal'):
        if orientation == 'horizontal':
            # Sort the contour points based on the x-coordinate (horizontal orientation)
            sorted_points = sorted(contour, key=lambda x: x[0][0])
        else:
            # Sort the contour points based on the y-coordinate (vertical orientation)
            sorted_points = sorted(contour, key=lambda x: x[0][1])

        # Take the last few points which should be at the end of the crack
        end_points = sorted_points[-10:]  # Taking last 10 points to average out any noise

        if orientation == 'horizontal':
            # Get min and max y-values to find height at the end of the crack
            min_y = min(end_points, key=lambda x: x[0][1])[0][1]
            max_y = max(end_points, key=lambda x: x[0][1])[0][1]
        else:
            # Get min and max x-values to find width at the end of the crack
            min_x = min(end_points, key=lambda x: x[0][0])[0][0]
            max_x = max(end_points, key=lambda x: x[0][0])[0][0]

        width_at_end = max_y - min_y if orientation == 'horizontal' else max_x - min_x
        return width_at_end

    def measure_contours(self, contours):
        # Initialize default values in case there are no contours
        features = {
            'length': 0,  # Default length when no contours are detected
            'width': 0,  # Default width when no contours are detected
            'angle': 0  # Default angle when no contours are detected
        }
        if contours:
            # Calculate arc lengths of all contours
            lengths = [cv2.arcLength(contour, True) for contour in contours]
            # Identify the longest contour
            longest_contour = max(contours, key=lambda x: cv2.arcLength(x, True))
            # Calculate the maximum length (if lengths are available)
            longest_length = max(lengths) if lengths else 0

            # Calculate the angle from the longest contour
            angle = self.angle_from_min_area_rect(longest_contour)
            # Calculate the width at the end of the longest contour
            width = self.find_crack_end_width(longest_contour)
            # Convert the width from pixels to centimeters
            width_cm = self.pixels_to_centimeters(width, 0, 2000, 34, 100)

            # Get the bounding rectangle to calculate length in cm
            x, y, w, h = cv2.boundingRect(longest_contour)
            end_x = x + w
            longest_length_cm = self.pixels_to_centimeters(end_x, 0, 2000, 34, 100) - 10

            # Update the features dictionary with actual measurements
            features.update({
                'length':  longest_length_cm,
                'width':     width_cm,
                'angle': angle
            })

        return features

    def convert_pixels_to_cm_width(self,width_in_pixels, total_pixels, physical_width_cm):
        """
        Converts a measurement from pixels to centimeters based on the total pixel width
        corresponding to a known physical width in centimeters.

        :param width_in_pixels: The width in pixels to be converted.
        :param total_pixels: Total number of pixels across the known physical width.
        :param physical_width_cm: The physical width in centimeters that corresponds to the total pixels.
        :return: The width in centimeters.
        """
        pixels_per_cm = total_pixels / physical_width_cm
        width_in_cm = width_in_pixels / pixels_per_cm
        return width_in_cm

    def pixels_to_centimeters(self,pixel_position, pixel_range_start, pixel_range_end, cm_range_start, cm_range_end):
        """
        Convert pixel position to centimeters in the real world.

        :param pixel_position: The pixel position to convert.
        :param pixel_range_start: The starting pixel of the measurable range.
        :param pixel_range_end: The ending pixel of the measurable range.
        :param cm_range_start: The starting centimeter mark in the real world.
        :param cm_range_end: The ending centimeter mark in the real world.
        :return: The equivalent centimeter position in the real world.
        """
        # Calculate the number of pixels per centimeter
        total_pixels = pixel_range_end - pixel_range_start
        total_cm = cm_range_end - cm_range_start
        pixels_per_cm = total_pixels / total_cm

        # Calculate the centimeter position corresponding to the given pixel position
        cm_position = cm_range_start + (pixel_position - pixel_range_start) / pixels_per_cm
        return cm_position


    def visualize_process(self):
        """Visualize each step of the process for understanding and debugging."""
        original = self.image
        gray_image = self.rotate_image(original, 0)  # Example: No rotation
        blurred = self.apply_gaussian_blur(gray_image)
        mag= self.detect_edges(blurred)
        processed_image = self.threshold_and_close(mag)

        plt.figure(figsize=(10, 8))
        plt.subplot(221); plt.imshow(original, cmap='gray'); plt.title('Original Image')
        plt.subplot(222); plt.imshow(blurred, cmap='gray'); plt.title('Blurred Image')
        plt.subplot(223); plt.imshow(mag, cmap='gray'); plt.title('Edge Enhanced Image')
        plt.subplot(224); plt.imshow(processed_image, cmap='gray'); plt.title('Thresholded and Closed Image')
        plt.tight_layout()
        plt.show()