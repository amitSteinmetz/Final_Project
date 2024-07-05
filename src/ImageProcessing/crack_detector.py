import cv2
import numpy as np
import math
import pandas as pd
import scipy.ndimage
from PIL import Image
from visualization import plot_image, plot_image_with_measurement
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


class CrackDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.fudgefactor = 5
        self.sigma = 200
        self.kernel = 2 * math.ceil(2 * self.sigma) + 1

    def orientated_non_max_suppression(self, mag, ang):
        ang_quant = np.round(ang / (np.pi / 4)) % 4
        winE = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        winSE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        winS = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        winSW = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

        magE = self.non_max_suppression(mag, winE)
        magSE = self.non_max_suppression(mag, winSE)
        magS = self.non_max_suppression(mag, winS)
        magSW = self.non_max_suppression(mag, winSW)

        mag[ang_quant == 0] = magE[ang_quant == 0]
        mag[ang_quant == 1] = magSE[ang_quant == 1]
        mag[ang_quant == 2] = magS[ang_quant == 2]
        mag[ang_quant == 3] = magSW[ang_quant == 3]
        return mag

    @staticmethod
    def non_max_suppression(data, win):
        data_max = scipy.ndimage.filters.maximum_filter(data, footprint=win, mode='constant')
        data_max[data != data_max] = 0
        return data_max



    def measure_crack(self, image_with_crack, crack_contour, real_worldW, real_worldH):
        if crack_contour is not None:
            # Draw the crack contour directly on the image
            cv2.drawContours(image_with_crack, [crack_contour], -1, (0, 255, 0), 3)  # Green color

            # Calculate the bounding rectangle for measurement purposes
            x, y, w, h = cv2.boundingRect(crack_contour)

            # Draw the bounding box around the crack in blue color
            cv2.rectangle(image_with_crack, (x, y), (x + w, y + h), (255, 0, 0), 2)

            crack_length_pixels = max(w, h)
            end_x = x + w
            end_y = y + h // 2

            # Optional: Draw measurement lines if needed
            line_length = 50  # Length of the line for showing measurements
            cv2.line(image_with_crack, (end_x, end_y), (end_x + line_length, end_y), (255, 0, 0), 2)  # Red line for length
            # Draw width measurement lines (vertical lines at the start and end of the crack)
            # cv2.line(image_with_crack, (x, y), (x, y + h), (255, 0, 0), 2)  # Start line
            # cv2.line(image_with_crack, (end_x, y), (end_x, y + h), (255, 0, 0), 2)  # End line

            rect = cv2.minAreaRect(crack_contour)
            box = cv2.boxPoints(rect)
            box = np.int_(box)
            cv2.drawContours(image_with_crack, [box], 0, (0, 0, 255), 2)  # Draw the rectangle in red
            # Identify bottom-left and top-right points

            top_left = min(box, key=lambda x: (x[1], x[0]))  # Smallest y, then smallest x for ties
            bottom_right = max(box, key=lambda x: (x[1], -x[0]))  # Largest y, then largest x for ties

            # Highlight these points
            cv2.circle(image_with_crack, tuple(top_left), 15, (255, 0, 0), -1)  # Blue circle at top-left
            cv2.circle(image_with_crack, tuple(bottom_right), 15, (255, 255, 0), -1)  # Yellow circle at bottom-right

            # Computing the angle using these points
            dx = bottom_right[0] - top_left[0]
            dy = bottom_right[1] - top_left[1]
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            # Calculate and display measurements
            angle, length, width = self.calculate_measurements(crack_contour, real_worldW, real_worldH)
            self.annotate_image(image_with_crack, x, y, angle, length, width)

            return crack_length_pixels, x, end_x, image_with_crack, length, width, angle
        else:
            return None, None, None, image_with_crack

    def calculate_crack_angle(self, crack_contour):
        if crack_contour is not None:
            rect = cv2.minAreaRect(crack_contour)
            width, height = rect[1]
            angle = rect[2]

            if width < height:
                # The angle is the angle of the width from the vertical axis
                angle += 90
            # Now, angle is the rotation from horizontal, where right is 0 degrees

            # Normalize angle to be within 0-180 degrees by adjusting from horizontal
            angle = (angle - 90) % 360  # This converts the angle from the horizontal

            if angle < 0:
                angle += 360  # Ensure angle is positive

            return angle
        else:
            return None


    def calculate_measurements(self, crack_contour, real_worldW, real_worldH):
        # Get the minimum area rectangle that bounds the crack
        rect = cv2.minAreaRect(crack_contour)
        box = cv2.boxPoints(rect)  # Get the four points of the bounding box
        box = np.int_(box)  # Convert points to integers

        width, height = rect[1]
        angle = self.calculate_crack_angle(crack_contour)

        # Determine which is width and which is height based on the angle
        if width < height:
            length = height
            width_at_ends = width
        else:
            length = width
            width_at_ends = height

        # Convert pixel dimensions to centimeters
        cm_length = self.pixels_to_centimeters(length, real_worldH, real_worldW, 2000)
        cm_length = round(cm_length + 34 - 10, 3) - 0.50  # Adjusting based on calibration

        cm_width_at_ends = self.pixels_to_centimeters(width_at_ends, real_worldH, real_worldW, 2000)
        cm_width_at_ends = round(cm_width_at_ends, 3)

        return angle, cm_length, cm_width_at_ends
    def annotate_image(self, image, x, y, angle, length, width):
        font_scale = 1.5
        vertical_gap = 60
        text_gap = 40
        cv2.putText(image, f'Length: {length:.2f} cm', (x, y - vertical_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 255), 2)
        cv2.putText(image, f'Width: {width:.2f} cm', (x, y - vertical_gap - text_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 255), 2)
        cv2.putText(image, f'Angle: {angle:.2f} deg', (x, y - vertical_gap - 2 * text_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)

    def pixels_to_centimeters(self, pixels, start_cm, end_cm, total_pixels):
        real_world_cm = end_cm - start_cm
        pixels_per_cm = total_pixels / real_world_cm
        return pixels / pixels_per_cm

    def detect_cracks(self):
        gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        gray_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        gray_image = np.uint8(gray_image)


        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)



        edge_enhanced = cv2.subtract(gray_image, blurred_image)
        sobelx = cv2.Sobel(edge_enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(edge_enhanced, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(sobelx, sobely)
        ang = np.arctan2(sobely, sobelx)
        mag = self.orientated_non_max_suppression(mag, ang)


        threshold = 4 * self.fudgefactor * np.mean(mag)
        mag[mag < threshold] = 0
        kernel = np.ones((5, 5), np.uint8)
        mag = cv2.morphologyEx(mag.astype(np.uint8), cv2.MORPH_CLOSE, kernel)



        contours, _ = cv2.findContours(mag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_width_pixels = gray_image.shape[1]
        possible_cracks = [cnt for cnt in contours if cv2.arcLength(cnt, False) > image_width_pixels * 0.9]
        possible_cracks = sorted(possible_cracks, key=lambda cnt: cv2.arcLength(cnt, False), reverse=True)

        return possible_cracks[0] if possible_cracks else None

    def draw_crack(self, image_with_crack, crack_contour):
        if len(image_with_crack.shape) == 2:
            image_with_crack = cv2.cvtColor(image_with_crack, cv2.COLOR_GRAY2BGR)
        if crack_contour is not None:
            cv2.drawContours(image_with_crack, [crack_contour], -1, (0, 0, 255), 3)
        return image_with_crack

    def convert_to_gray(self, image):
        if image.mode != 'L':
            return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        else:
            return np.array(image)

    def load_image(self, image_path):
        return Image.open(image_path)

    def process_and_draw_cracks(self, df,num_img):
        contours = self.detect_cracks()
        original_image = cv2.imread(self.image_path)
        image = self.load_image(self.image_path)
        gray_image_np = self.convert_to_gray(image)


        if original_image is not None:
            crack_length_pixels, x, end_x, image_with_measurement, lenOfCrack, lenOfWid, angle = self.measure_crack(
                original_image, contours, 100, 34)
            new_row = {'N': num_img, 'a': lenOfCrack, 'w': lenOfWid, 'z': angle}


            if not pd.DataFrame([new_row]).isnull().all(axis=1).any():
                if not df.empty:
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                else:
                    df = pd.DataFrame([new_row])

            plot_image(np.array(image), original_image, title=f"Image with Crack Contour number {num_img}")
            if crack_length_pixels is not None:
                 title = f"Image with Crack Contour number {num_img}"
                 plot_image_with_measurement(gray_image_np, image_with_measurement, crack_length_pixels, x, end_x,
                                             title)



        return original_image, df