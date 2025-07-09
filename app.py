
import flask
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

# Initialize Flask app
app = Flask(__name__)

# Define the path for the model and scaler files
MODEL_PATH = 'logistic_regression_model.pkl'
SCALER_PATH = 'scaler.pkl'
SELECTED_FEATURES_PATH = 'selected_features.pkl'
CATEGORICAL_COLS_PATH = 'categorical_cols.pkl'
NUMERICAL_COLS_PATH = 'numerical_cols.pkl'

# Function to load the model and preprocessing objects
def load_model_and_preprocessing():
    """Loads the trained model and preprocessing objects from disk."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        selected_features = joblib.load(SELECTED_FEATURES_PATH)
        categorical_cols = joblib.load(CATEGORICAL_COLS_PATH)
        numerical_cols = joblib.load(NUMERICAL_COLS_PATH)
        return model, scaler, selected_features, categorical_cols, numerical_cols
    except FileNotFoundError:
        return None, None, None, None, None

# Load the model and preprocessing objects
model, scaler, selected_features, categorical_cols, numerical_cols = load_model_and_preprocessing()

# Check if loading was successful
if model is None:
    print("Error: Model or preprocessing objects not found. Please ensure the model training and saving steps were completed.")
    # In a real app, you might want to handle this more gracefully,
    # perhaps by rendering an error page or exiting.
else:
    print("🚀Model and preprocessing objects loaded successfully for the web app.")


# Define the home route
@app.route('/')
def home():
    """Renders the home page with the risk prediction form."""
    if selected_features is None:
        return "Error: Selected features not loaded.", 500
    return render_template('index.html', features=selected_features)

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission, preprocesses input, and returns prediction."""
    if model is None or scaler is None or selected_features is None or categorical_cols is None or numerical_cols is None:
        return "Error: Model or preprocessing objects not loaded.", 500

    try:
        # Get user input from the form
        user_input = {}
        for feature in selected_features:
            user_input[feature] = request.form.get(feature)

        # Convert input to appropriate types (handle potential errors)
        processed_input = {}
        for col in numerical_cols:
             # Use .get() with a default value and convert to float
             processed_input[col] = float(user_input.get(col, 0) or 0)
        for col in categorical_cols:
             # Use .get() with a default value and convert to int
             processed_input[col] = int(user_input.get(col, 0) or 0)


        user_df = pd.DataFrame([processed_input])

        # Preprocess the user input
        user_numerical = user_df[numerical_cols]
        user_categorical = user_df[categorical_cols]

        user_categorical_encoded = pd.get_dummies(user_categorical, columns=categorical_cols, drop_first=True)
        user_numerical_scaled = scaler.transform(user_numerical)
        user_numerical_scaled = pd.DataFrame(user_numerical_scaled, columns=numerical_cols)

        user_processed = pd.concat([user_numerical_scaled, user_categorical_encoded], axis=1)

        # Ensure all selected features from training are present, add missing ones with 0
        for col in selected_features:
            if col not in user_processed.columns:
                user_processed[col] = 0

        # Reorder columns to match the training data's feature order
        user_processed = user_processed[selected_features]


        # Make prediction
        prediction = model.predict(user_processed)[0]

        # Map prediction to risk level (assuming binary target 0 or 1)
        risk_level = "High Risk" if prediction == 1 else "Low Risk"

        # Note: For a multi-class target (Low, Medium, High), you would need to map
        # the model's output (e.g., predicted class or probabilities) to these zones.
        # This example assumes a binary target mapped to Low/High for simplicity.


        return render_template('result.html', prediction=risk_level)

    except Exception as e:
        return f"An error occurred: {e}", 500

# Run the Flask app
if __name__ == '__main__':
    print("Started application🚀")
    # This will run the Flask development server
    # debug=True allows for automatic code reloading and detailed error messages
    app.run(debug=True)
