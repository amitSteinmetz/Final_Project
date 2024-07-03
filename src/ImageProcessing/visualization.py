import os

import matplotlib.pyplot as plt


def plot_image(original_image, image_with_crack, title="Image with Crack", save_directory="ImgResult"):
    # Ensure the save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plt.figure(figsize=(20, 10))
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')
    plt.title("Original Image")

    # Image with Crack
    plt.subplot(1, 2, 2)
    plt.imshow(image_with_crack, cmap='gray')
    plt.axis('off')
    plt.title(title)

    # Construct the file path
    save_path = os.path.join(save_directory, f"{title}.png")
    # Save the plot
    plt.savefig(save_path)
    # Close the figure to free up memory
    plt.pause(1)  # Display for 1 second
    plt.close()

# def plot_image_with_measurement(gray_image, image_with_measurement, crack_length_pixels, x, end_x):
#     plt.figure(figsize=(10, 5))
#     plt.imshow(image_with_measurement)
#     plt.axvline(x=x, color='r', linestyle='--')
#     plt.axvline(x=end_x, color='r', linestyle='--')
#     plt.title(f"Crack Length in Pixels: {crack_length_pixels}")
#     plt.axis('off')
#     plt.show()
def pixels_to_centimeters(pixels, start_cm, end_cm, total_pixels):
    real_world_cm = end_cm - start_cm
    pixels_per_cm = total_pixels / real_world_cm
    return pixels / pixels_per_cm


def plot_image_with_measurement(original_image, image_with_measurement, crack_length_pixels, x, end_x, title):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(image_with_measurement, cmap='gray')
    plt.axvline(x=x, color='r', linestyle='--')
    plt.axvline(x=end_x, color='r', linestyle='--')
    centimeters = pixels_to_centimeters(crack_length_pixels, 34, 100, 2000)
    plt.title(f"{title} Length in CM: {round((centimeters + 34 - 10)-0.50, 3)}, Scale (cm/unit): ")

    plt.show(block=False)  # Display the plot without blocking the rest of the script
    plt.pause(3)           # Keep the plot open for 3 seconds
    plt.close()            # Close the plot automatically

# Ensure to call this function appropriately in your script where it fits your flow.