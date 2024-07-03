import glob
import os
import cv2
import numpy as np
import math
import scipy.ndimage
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
from visualization import plot_image, plot_image_with_measurement
# Define the CrackDetector class with additional methods to draw cracks and measure them


class CrackDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.with_nmsup = True
        self.fudgefactor = 4
        self.sigma = 30
        self.kernel = 2 * math.ceil(2 * self.sigma) + 1
        # self.df = pd.DataFrame(columns=['N', 'a', 'w', 'z'])
        self.num_img = 1

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

    def non_max_suppression(self, data, win):
        data_max = scipy.ndimage.filters.maximum_filter(data, footprint=win, mode='constant')
        data_max[data != data_max] = 0
        return data_max

    def measure_crack(self,image_with_crack, crack_contour, real_worldW, real_worldH):
        if crack_contour is not None:
            x, y, w, h = cv2.boundingRect(crack_contour)
            crack_length_pixels = max(w, h)

            # Draw length measurement line
            end_x = x + w
            end_y = y + h // 2
            line_length = 50  # Adjust if necessary
            cv2.line(image_with_crack, (end_x, end_y), (end_x + line_length, end_y), (0, 255, 0), 2)

            # Draw width measurement lines (vertical lines at the start and end of the crack)
            cv2.line(image_with_crack, (x, y), (x, y + h), (255, 0, 0), 2)  # Start line
            cv2.line(image_with_crack, (end_x, y), (end_x, y + h), (255, 0, 0), 2)  # End line

            rect = cv2.minAreaRect(crack_contour)
            box = cv2.boxPoints(rect)
            box = np.int_(box)

            cv2.drawContours(image_with_crack, [box], 0, (0, 0, 255), 2)  # Draw the rectangle in red

            # The angle from the rect is the angle the rectangle rotates clockwise about its center
            # OpenCV defines the angle as the rotation needed such that the rectangle width is greater than the height
            width, height = rect[1][0], rect[1][1]
            angle = rect[2]

            if width < height:
                angle = 90 + angle  # Adjust angle for vertical orientation
            else:
                # For horizontal orientation, no adjustment needed, but you might want to normalize it
                if angle < -45:
                    angle = angle + 90

            # Optionally, convert angle to a range of [0, 180) for uniformity
            angle = (angle + 180) % 180

            # print(f"Adjusted Angle: {angle} degrees")
            # Increase text size by adjusting the font scale parameter
            font_scale = 1.0  # Adjust font scale if necessary

            # Adjust the vertical position of the text to move it further above the crack
            # and increase the gap between the two lines of text significantly
            vertical_gap = 50  # Gap from the top of the crack to the first line of text
            text_gap = 40  # Additional gap between the two lines of text
            centimetersCrack = self.pixels_to_centimeters(crack_length_pixels, real_worldH, real_worldW, 2000)
            lenOfCrack = round(centimetersCrack + 34-10, 3)
            centimetersWidth = self.pixels_to_centimeters(min(w, h), real_worldH, real_worldW, 2000)
            lenOfWid = round(centimetersWidth, 3)
            # Adjust the vertical positions
            cv2.putText(image_with_crack, f'Len: {round(centimetersCrack + 34, 3)}cm', (x, y - vertical_gap),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
            # Increase the gap for the 'Wid' text by adding 'text_gap' to the previous vertical offset
            cv2.putText(image_with_crack, f'Wid: {round(centimetersWidth, 3)}cm', (x, y - vertical_gap - text_gap),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
            # Annotate the angle
            cv2.putText(image_with_crack, f'Ang: {angle:.2f} deg', (x, y - vertical_gap - 2 * text_gap),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
            return crack_length_pixels, x, end_x, image_with_crack, lenOfCrack, lenOfWid, angle
        else:
            return None, None, None, image_with_crack

    def pixels_to_centimeters(self,pixels, start_cm, end_cm, total_pixels):
        real_world_cm = end_cm - start_cm
        pixels_per_cm = total_pixels / real_world_cm
        return pixels / pixels_per_cm

    def detect_cracks(self):

        # Read the image in grayscale as 8-bit image
        gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        # Optionally normalize and convert back to 8-bit
        gray_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        gray_image = np.uint8(gray_image)

        # Apply Gaussian blur to reduce noise and improve edge detection
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Subtract the blurred image from the original to enhance edges
        edge_enhanced = cv2.subtract(gray_image, blurred_image)

        # Apply Sobel filter to get the derivative in x and y direction
        sobelx = cv2.Sobel(edge_enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(edge_enhanced, cv2.CV_64F, 0, 1, ksize=3)

        # Compute the magnitude of the gradient
        mag = np.hypot(sobelx, sobely)

        # Compute the orientation of the gradient
        ang = np.arctan2(sobely, sobelx)

        # Apply non-maximum suppression with orientation
        mag = self.orientated_non_max_suppression(mag, ang)

        # Set threshold to identify significant gradients
        threshold = 4 * self.fudgefactor * np.mean(mag)
        mag[mag < threshold] = 0

        # Apply morphological closing to the thresholded image
        kernel = np.ones((5, 5), np.uint8)  # You can adjust the size of the kernel
        mag = cv2.morphologyEx(mag.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # Find contours from the edges detected
        contours, _ = cv2.findContours(mag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # The width of the image in pixels
        image_width_pixels = gray_image.shape[1]

        # We'll assume that a contour that spans almost the entire width of the image is the crack
        possible_cracks = [cnt for cnt in contours if cv2.arcLength(cnt, False) > image_width_pixels * 0.9]

        # Sort possible cracks by their arc length to find the longest one
        possible_cracks = sorted(possible_cracks, key=lambda cnt: cv2.arcLength(cnt, False), reverse=True)

        # Take the longest contour as the crack
        crack_contour = possible_cracks[0] if possible_cracks else None

        # Draw the contour on the image
        image_with_crack = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        if crack_contour is not None:
            cv2.drawContours(image_with_crack, [crack_contour], -1, (255, 0, 0), 3)

        # If a crack contour is found, find the bounding box and calculate its length
        if crack_contour is not None:
            # Find the bounding box of the contour to get the pixel length
            x, y, w, h = cv2.boundingRect(crack_contour)
            # Use the larger of the width or height as the crack's length in pixels
            crack_length_pixels = max(w, h)

            # Display the image with the contour and measurements
            # plt.figure(figsize=(10, 5))
            # plt.imshow(image_with_crack)
            # plt.axvline(x=x, color='r', linestyle='--')  # Vertical line at the start of the crack
            # plt.axvline(x=x + w, color='r', linestyle='--')  # Vertical line at the end of the crack
            # plt.title(f"Crack Length in Pixels: {crack_length_pixels}")
            # plt.axis('off')
            # plt.show()
        else:
            crack_length_pixels = None

        # Output the crack length in pixels
        return crack_contour

    def save_image_with_crack(self,original_image, image_with_crack, title, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        save_path = os.path.join(save_directory, f"{title}.png")
        # Ensure original_image is in BGR if it was converted to grayscale and back
        if len(original_image.shape) == 2 or original_image.shape[2] == 1:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        if len(image_with_crack.shape) == 2 or image_with_crack.shape[2] == 1:
            image_with_crack = cv2.cvtColor(image_with_crack, cv2.COLOR_GRAY2BGR)
        # Combine original and processed images side by side for comparison
        combined_image = np.concatenate((original_image, image_with_crack), axis=1)
        # Save the combined image
        cv2.imwrite(save_path, combined_image)
    def draw_crack(self, image_with_crack, crack_contour):
        if len(image_with_crack.shape) == 2:
            image_with_crack = cv2.cvtColor(image_with_crack, cv2.COLOR_GRAY2BGR)
        if crack_contour is not None:
            cv2.drawContours(image_with_crack, [crack_contour], -1, (0, 0, 255), 3)
        return image_with_crack

    def convert_to_gray(self,image):
        if image.mode != 'L':
            return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        else:
            return np.array(image)

    def load_image(self,image_path):
        return Image.open(image_path)

    def process_and_draw_cracks(self, df):
        contours = self.detect_cracks()

        original_image = cv2.imread(self.image_path)
        image = self.load_image(self.image_path)
        gray_image_np = self.convert_to_gray(image)
        num_img = 1

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
                plot_image_with_measurement(gray_image_np, image_with_measurement, crack_length_pixels, x, end_x)

        num_img += 1

        return original_image, df

def execute_crack_detection(image_path, df):

    detector = CrackDetector(image_path)
    image_with_cracks, df = detector.process_and_draw_cracks(df)

    return image_with_cracks, df

df = pd.DataFrame(columns=['N', 'a', 'w', 'z'])
image_path = "../../data/0_0/0_0-2/2.1/0_0-2,1_1/0_0-2,1_1_5507_20210429_16_15_36,629.jpg"
image_directory = "../../data/0_0/0_0-2/2.1/0_0-2,1_1"
image_paths = sorted(glob.glob(os.path.join(image_directory, '*.jpg')))
for image_path in image_paths:
    result_image_with_cracks,df = execute_crack_detection(image_path,df)

    # result_image_with_cracks = cv2.cvtColor(result_image_with_cracks, cv2.COLOR_BGR2RGB)
    # plt.imshow(result_image_with_cracks)
    # plt.axis('off')  # Hide the axis
    # plt.show()
df.to_csv('output_filename.csv', index=False)
df.to_excel('output_filename.xlsx', index=False, engine='openpyxl')
print(df)
