# validation.py
import tensorflow as tf
from data_preprocessing import load_data_from_file

def evaluate_model(model_path, file_path):
    # Load the model with custom_objects to pass the loss function
    model =  tf.keras.models.load_model('model.keras')

    # Load the validation dataset (same process as in training)
    val_dataset = load_data_from_file(file_path, batch_size=1, sequence_length=5)

    # Evaluate the model on validation data
    results = model.evaluate(val_dataset)
    print(f"Validation Loss: {results[0]}, Validation MAE: {results[1]}")

if _name_ == '_main_':
    evaluate_model('crack_progression_model.h5', './crack_data.csv')