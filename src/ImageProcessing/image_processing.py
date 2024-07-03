import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


def load_image(image_path):
    return Image.open(image_path)

def convert_to_gray(image):
    if image.mode != 'L':
        return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    else:
        return np.array(image)


# Function to detect the ruler
def detect_ruler_and_cracks(image_gray):
    # Apply Canny edge detection
    edges = cv2.Canny(image_gray, 100, 200)

    # Dilate the edge lines for better detection
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Detect lines using Hough Transform for ruler detection
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)

    # Initialize an empty image to draw lines
    line_image = np.zeros_like(image_gray)

    # Find contours for crack detection
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out the contours that correspond to ruler lines and detect potential cracks
    potential_cracks = [cnt for cnt in contours if cv2.contourArea(cnt) < 1000]  # Example threshold value

    # Draw potential cracks on the line image in a different color
    cv2.drawContours(line_image, potential_cracks, -1, (255, 0, 0), 1)  # Blue color for cracks

    # If lines are detected, draw them on the line image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw the line
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 3)  # White color for ruler lines

    # Show the original image with detected lines and potential cracks
    plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_GRAY2RGB))
    plt.title('Detected Ruler Lines and Potential Cracks')
    plt.show()

    return line_image


def visualize_contours(image, edges):
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank canvas with the same shape as the original image
    canvas = np.zeros_like(image)

    # Draw all contours on the canvas in green
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

    # Overlay contours on the original image
    result = cv2.addWeighted(image, 0.7, canvas, 0.3, 0)

    # Display the result
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.show()



def detect_crack(image_gray):
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # Apply Canny edge detection to find edges
    edges = cv2.Canny(blurred_image, 50, 150)

    # Find contours in the edge-detected image to identify continuous lines that could be cracks
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # The width of the image in pixels
    image_width_pixels = image_gray.shape[1]

    # We'll assume that a contour that spans almost the entire width of the image is the crack
    possible_cracks = [cnt for cnt in contours if cv2.arcLength(cnt, False) > image_width_pixels * 0.9]

    # Sort possible cracks by their arc length to find the longest one
    possible_cracks = sorted(possible_cracks, key=lambda cnt: cv2.arcLength(cnt, False), reverse=True)

    # Take the longest contour as the crack
    crack_contour = possible_cracks[0] if possible_cracks else None

    # Draw the contour on the image
    image_with_crack = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    if crack_contour is not None:
        cv2.drawContours(image_with_crack, [crack_contour], -1, (255, 0, 0), 3)

    # If a crack contour is found, find the bounding box and calculate its length
    if crack_contour is not None:
        # Find the bounding box of the contour to get the pixel length
        x, y, w, h = cv2.boundingRect(crack_contour)
        # Use the larger of the width or height as the crack's length in pixels
        crack_length_pixels = max(w, h)

        # Display the image with the contour and measurements
        plt.figure(figsize=(10, 5))
        plt.imshow(image_with_crack)
        plt.axvline(x=x, color='r', linestyle='--')  # Vertical line at the start of the crack
        plt.axvline(x=x + w, color='r', linestyle='--')  # Vertical line at the end of the crack
        plt.title(f"Crack Length in Pixels: {crack_length_pixels}")
        plt.axis('off')
        plt.show()
    else:
        crack_length_pixels = None

    # Output the crack length in pixels
    return crack_contour

def draw_crack(image_with_crack, crack_contour):
    if len(image_with_crack.shape) == 2:
        image_with_crack = cv2.cvtColor(image_with_crack, cv2.COLOR_GRAY2BGR)
    if crack_contour is not None:
        cv2.drawContours(image_with_crack, [crack_contour], -1, (0, 255, 0), 3)
        x, y, w, h = cv2.boundingRect(crack_contour)
        cv2.rectangle(image_with_crack, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image_with_crack

def measure_crack(image_with_crack, crack_contour,real_worldW,real_worldH):
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
        box = np.int0(box)
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

        #print(f"Adjusted Angle: {angle} degrees")
        # Increase text size by adjusting the font scale parameter
        font_scale = 1.0  # Adjust font scale if necessary

        # Adjust the vertical position of the text to move it further above the crack
        # and increase the gap between the two lines of text significantly
        vertical_gap = 50  # Gap from the top of the crack to the first line of text
        text_gap = 40  # Additional gap between the two lines of text
        centimetersCrack = pixels_to_centimeters(crack_length_pixels, real_worldH, real_worldW, 2000)
        lenOfCrack =round(centimetersCrack +34,3)
        centimetersWidth = pixels_to_centimeters(min(w, h), real_worldH, real_worldW, 2000)
        lenOfWid = round(centimetersWidth,3)
        # Adjust the vertical positions
        cv2.putText(image_with_crack, f'Len: {round(centimetersCrack +34,3)}cm', (x, y - vertical_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        # Increase the gap for the 'Wid' text by adding 'text_gap' to the previous vertical offset
        cv2.putText(image_with_crack, f'Wid: {round(centimetersWidth,3)}cm', (x, y - vertical_gap - text_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        # Annotate the angle
        cv2.putText(image_with_crack, f'Ang: {angle:.2f} deg', (x, y - vertical_gap - 2 * text_gap),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        return crack_length_pixels, x, end_x, image_with_crack ,lenOfCrack ,lenOfWid, angle
    else:
        return None, None, None, image_with_crack


def pixels_to_centimeters(pixels, start_cm, end_cm, total_pixels):
    real_world_cm = end_cm - start_cm
    pixels_per_cm = total_pixels / real_world_cm
    return pixels / pixels_per_cm
