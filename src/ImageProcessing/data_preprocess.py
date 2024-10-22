import os
import glob
import cv2
import pandas as pd
import tensorflow as tf
import numpy as np
from test2 import CrackDetector  # Assuming CrackDetector and related functions are defined here

# Enable memory growth for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def preprocess_image(image_path):
    """Load and preprocess the image."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)  # Assuming grayscale images
    image = tf.image.resize(image, [2048, 2048])  # Keep original size
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
    return image

def setup_dataset(df, batch_size=2):
    """Create a TensorFlow dataset prepared for training, including features."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame, but got {}".format(type(df)))

    required_columns = ['image_path', 'a', 'w', 'z']
    if not all(col in df.columns for col in required_columns):
        raise KeyError(f"DataFrame is missing one or more required columns: {required_columns}")

    image_paths = df['image_path'].tolist()
    features = df[['a', 'w', 'z']].astype(np.float32).to_numpy()

    path_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    feature_dataset = tf.data.Dataset.from_tensor_slices(features)

    image_dataset = path_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    final_dataset = tf.data.Dataset.zip((image_dataset, feature_dataset))

    final_dataset = final_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return final_dataset

def create_dataframe(image_paths, n_values):
    columns = ['N', 'a', 'w', 'z', 'image_path']
    rows = []

    for idx, image_path in enumerate(image_paths):
        if idx >= len(n_values):
            break
        row = {'N': n_values[idx], 'a': None, 'w': None, 'z': None, 'image_path': image_path}
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    return df

def process_images_and_update_df(df):
    """Process images using CrackDetector and update DataFrame with detected features."""
    for index, row in df.iterrows():
        image_path = row['image_path']
        try:
            # Assume CrackDetector exists and processes the image
            detector = CrackDetector(image_path)
            contours = detector.detect_cracks()
            features = detector.measure_contours(contours)

            df.at[index, 'a'] = features['length']
            df.at[index, 'w'] = features['width']
            df.at[index, 'z'] = features['angle']

            image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_data is None:
                raise FileNotFoundError(f"Image at {image_path} could not be loaded.")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    return df

def split_dataset(dataset, train_fraction=0.8):
    total_batches = len(list(dataset))
    train_batches = int(total_batches * train_fraction)
    validation_batches = total_batches - train_batches

    train_dataset = dataset.skip(validation_batches)
    val_dataset = dataset.take(validation_batches)

    return train_dataset, val_dataset

def prepare_dataset(df, batch_size=10):
    """Create a TensorFlow dataset from the DataFrame."""
    features = df[['a', 'w', 'z']].astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(features.to_numpy())
    return dataset.batch(batch_size)

# def preprocess_image(image_path):
#     """Load and preprocess the image at original size."""
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image, channels=1)  # Assuming grayscale
#     image = tf.image.resize(image, [512, 512])  # Reducing size to manage resource usage
#     image = image / 255.0  # Normalize pixel values to be between 0 and 1
#     return image

def load_data_from_file(file_path, batch_size=1, sequence_length=5):
    """
    Load the dataset from a CSV or Excel file and prepare it for training.

    Args:
    - file_path: Path to the CSV/Excel file containing crack features.
    - batch_size: Batch size for training the model.
    - sequence_length: The number of time steps (sequential entries) for time-series predictions.

    Returns:
    - A TensorFlow dataset ready for training.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("File format not supported. Please provide a CSV or Excel file.")

    # Ensure the DataFrame contains necessary columns
    required_columns = ['a', 'w', 'z']
    if not all(col in df.columns for col in required_columns):
        raise KeyError(f"DataFrame is missing required columns: {required_columns}")

    # Convert features into numpy array
    features = df[required_columns].astype('float32').to_numpy()

    # Prepare sequences for time-series training (e.g., 5 time steps per sample)
    sequences, labels = [], []
    for i in range(len(features) - sequence_length):
        sequences.append(features[i:i + sequence_length])  # Input sequence
        labels.append(features[i + sequence_length])  # Label is the next time step

    # Convert to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((sequences, labels))

    # Batch and prefetch for performance
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
def load_data(image_paths, labels, sequence_length=5):
    """Create a dataset from slices of image paths and their corresponding labels."""
    def gen():
        for i in range(len(image_paths) - sequence_length + 1):
            sequence_paths = image_paths[i:i + sequence_length]
            sequence_labels = labels.iloc[i:i + sequence_length].to_numpy()
            yield np.array([preprocess_image(p) for p in sequence_paths]), sequence_labels

    output_signature = (
        tf.TensorSpec(shape=(sequence_length, 512, 512, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(sequence_length, 3), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature
    )
    return dataset.batch(10)


def load_data_from_excel(excel_file_path, sequence_length=5):
    """Load dataset from Excel, prepare sequences and return a TensorFlow dataset."""
    new_df = pd.read_excel(excel_file_path)
    image_paths = new_df['image_path'].tolist()
    labels = new_df[['a', 'w', 'z']]  # Assuming 'a', 'w', 'z' are columns in the Excel for labels
    train_dataset = load_data(image_paths, labels, sequence_length)
    return train_dataset
# def preprocess_image(image_path):
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image, channels=1)  # Assuming grayscale images
#     image = tf.image.resize(image, [1048, 1048])
#     image = image / 255.0
#     return image
#
# def load_data(df):
#     image_paths = df['image_path'].tolist()
#     labels = df[['a', 'w', 'z']]  # Ensure these columns exist in your DataFrame
#
#     # Create a dataset from tensor slices
#     dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels.to_numpy()))
#
#     # Map the dataset to preprocess images and format labels correctly
#     dataset = dataset.map(lambda x, y: (preprocess_image(x), y))
#     return dataset
def save_and_load_data(df):
    # Save DataFrame to CSV and Excel
    df.to_csv('crack_data.csv', index=False)
    df.to_excel('crack_data.xlsx', index=False)

    # Reload and preprocess data for training
    new_df = pd.read_csv('crack_data.csv')
    train_dataset = load_data(new_df)  # Ensure this function is set up to handle a DataFrame directly
    return train_dataset




def create_dataframe( image_paths, n_values):
    columns = ['N', 'a', 'w', 'z', 'image_path']
    rows = []


    for idx, image_path in enumerate(image_paths):
        if idx >= len(n_values):  # Check to avoid index error
            break
        # Prepare each row as a dictionary
        row = {'N': n_values[idx], 'a': None, 'w': None, 'z': None, 'image_path': image_path}
        rows.append(row)

    # Create DataFrame from rows
    df = pd.DataFrame(rows, columns=columns)

    return df


def generate_n_values(start=100, end=30000, increments=[100, 500, 1000], probs=[0.18, 0.11, 0.71]):
    """
    Generates 'N' values with variable increments based on probabilities.
    - start: Starting value of 'N'.
    - end: Maximum value of 'N' to not exceed.
    - increments: Possible increments of 'N'.
    - probs: Probabilities associated with each increment.
    """
    n_values = [start]
    current = start
    while current < end:
        inc = np.random.choice(increments, p=probs)
        next_value = current + inc
        if next_value > end:
            break
        n_values.append(next_value)
        current = next_value
    return n_values


def process_images(df):
    for index, row in df.iterrows():
        image_path = row['image_path']
        try:
            detector = CrackDetector(image_path)
            contours = detector.detect_cracks()
            features = detector.measure_contours(contours)  # Assume this returns dict with 'length', 'width', 'angle'

            # Update DataFrame directly with features
            df.at[index, 'a'] = features['length']
            df.at[index, 'w'] = features['width']
            df.at[index, 'z'] = features['angle']

            # Visualization (if necessary)
            # if contours:
            #     detector.visualize_longest_contour(detector.image, contours)

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    return df

def generate_n_sequence(start=100, total_values=348):
    """
    Generates an "N" sequence based on the observed increments pattern.

    :param start: The starting value of the sequence.
    :param total_values: The total number of values to generate.
    :return: A list containing the generated sequence of "N" values.
    """
    n_values = [start]
    current_value = start

    for i in range(1, total_values):
        if i < 50:
            increment = 100
        elif i < 80:
            increment = 500
        else:
            increment = 1000
        current_value += increment
        n_values.append(current_value)

    return n_values