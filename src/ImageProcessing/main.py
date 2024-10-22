import glob
import os
import cv2
import numpy as np
import math
import scipy.ndimage
import matplotlib.pyplot as plt
from data_preprocessing import setup_dataset, process_images_and_update_df,create_dataframe,load_data_from_file, process_images,generate_n_sequence,save_and_load_data,load_data_from_excel,prepare_dataset
from model import  train_and_validate_model

# from test2 import CrackDetector  # Assuming CrackDetector and related functions are defined here


# def generate_n_sequence(start=100, total_values=348):
#     """
#     Generates an "N" sequence based on the observed increments pattern.
#
#     :param start: The starting value of the sequence.
#     :param total_values: The total number of values to generate.
#     :return: A list containing the generated sequence of "N" values.
#     """
#     n_values = [start]
#     current_value = start
#
#     for i in range(1, total_values):
#         if i < 50:
#             increment = 100
#         elif i < 80:
#             increment = 500
#         else:
#             increment = 1000
#         current_value += increment
#         n_values.append(current_value)
#
#     return n_values

def split_dataset(dataset, train_fraction=0.5):  # Updated to use 50% for training
    """
    Splits the dataset into training and validation datasets.
    Args:
    - dataset: TensorFlow Dataset object.
    - train_fraction: Fraction of the dataset to use for training.

    Returns:
    - train_dataset: Dataset for training.
    - val_dataset: Dataset for validation.
    """
    # Total number of elements in the dataset
    total_batches = len(list(dataset))

    # Determine the number of batches needed for the training set
    train_batches = int(total_batches * train_fraction)
    validation_batches = total_batches - train_batches  # Rest for validation

    # Split the dataset
    train_dataset = dataset.skip(validation_batches)
    val_dataset = dataset.take(validation_batches)

    return train_dataset, val_dataset


if _name_ == '_main_':
    # image_directory = "../data/0_0/0_0-2/2.1/0_0-2,1_1"
    n_values = generate_n_sequence(100, 348)  # Generate 'N' values
    # df = create_dataframe(image_directory, n_values)  # Create DataFrame
    # df = process_images(df)  # Process images and update DataFrame

    image_directory = "../data/0_0/0_0-2/2.1/0_0-2,1_1"
    image_paths = sorted(glob.glob(os.path.join(image_directory, '*.jpg')))
    # image_paths = image_paths[:2]
    # Process images and update DataFrame with features
    df = create_dataframe(image_paths, n_values)

    # Process images and update DataFrame
    df = process_images_and_update_df(df)
    # Optionally save the updated DataFrame
    df.to_csv('updated_features.csv', index=False)

    # Prepare the dataset for training
    # dataset = setup_dataset(df, batch_size=2)

    dataset = load_data_from_file('./updated_features.csv', batch_size=2, sequence_length=5)
    # Split the dataset into training and validation sets
    train_ds, val_ds = split_dataset(dataset, train_fraction=0.8)

    # Train and validate the model
    model, history = train_and_validate_model(train_ds, val_ds, epochs=20)
    # Train and validate the model
    # model, history = train_and_validate_model(train_ds, validation_ds)

    # model = train_and_validate_model(train_ds, validation_ds)
