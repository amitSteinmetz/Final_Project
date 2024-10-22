# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer

def create_lstm_model(sequence_length, feature_size):
    """
    Create an LSTM model for predicting crack length, width, and angle.

    Args:
    - sequence_length: The number of time steps in the input sequence.
    - feature_size: The number of features per time step (e.g., length, width, angle).

    Returns:
    - A compiled LSTM model.
    """
    model = Sequential([
        InputLayer(input_shape=(sequence_length, feature_size)),
        LSTM(64, activation='relu'),
        Dense(3)  # Output layer: predicting length (a), width (w), and angle (z)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_validate_model(train_ds, val_ds, epochs=20):
    """Train and validate the model."""
    # Assuming a sequence length of 5 and 3 features (a, w, z)
    model = create_lstm_model(sequence_length=5, feature_size=3)
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
    model.save('model.keras')
    return model, history