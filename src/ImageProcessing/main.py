import os
import glob
import pandas as pd
from crack_detector import CrackDetector
from visualization import plot_image, plot_image_with_measurement
import numpy as np


def execute_crack_detection(image_path, df, num_img):
    detector = CrackDetector(image_path)
    image_with_cracks, df = detector.process_and_draw_cracks(df, num_img)

    return image_with_cracks, df


def main():
    df = pd.DataFrame(columns=['N', 'a', 'w', 'z'])
    image_directory = r"C:\Users\Owner\PycharmProjects\Final_Project\data\0_0\0_0-2\2.1\0_0-2,1_1"
    image_paths = sorted(glob.glob(os.path.join(image_directory, '*.jpg')))
    num_img=2

    for image_path in image_paths:
        detector = CrackDetector(image_path)
        image = detector.load_image(image_path)
        result_image_with_cracks, df = execute_crack_detection(image_path, df,num_img)
        num_img+=1

        # Visualization functions
      #  plot_image(np.array(image), result_image_with_cracks, title="Detected Crack")

    df.to_csv('output_filename.csv', index=False)
    df.to_excel('output_filename.xlsx', index=False, engine='openpyxl')
    print(df)

if __name__ == "__main__":
    main()
