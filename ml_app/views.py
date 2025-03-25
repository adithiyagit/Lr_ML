import pandas as pd
import numpy as np
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
# Full path to the dataset file
DATASET_PATH = 'manual_dataset.csv'  # Update this path if needed

# Path to the model file
MODEL_PATH = 'logistic_regression_model.pkl'

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Check if the dataset file exists
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset file not found at: {DATASET_PATH}")
else:
    print(f"Dataset file found at: {DATASET_PATH}")

# Check if the model file exists, and if not, train and save the model
if not os.path.exists(MODEL_PATH):
    # Load the dataset
    dataset = pd.read_csv(DATASET_PATH)

    # Split features and target
    X = dataset[['Age', 'Income']]
    y = dataset['Purchased']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the model and scaler to a file
    joblib.dump((model, sc), MODEL_PATH)
    print("Model trained and saved as 'logistic_regression_model.pkl'")
else:
    # Load the model and scaler from the file
    model, sc = joblib.load(MODEL_PATH)
    print("Model loaded from 'logistic_regression_model.pkl'")

# View to handle the prediction form and result

@csrf_exempt
def predict_survival(request):
    if request.method == 'POST':
        # Get user input from the form
        age = float(request.POST['age'])
        income = float(request.POST['income'])

        # Scale the input data using the saved scaler
        input_data = sc.transform(np.array([[age, income]]))

        # Make prediction
        prediction = model.predict(input_data)

        # Render the result page with the prediction
        return render(request, 'result.html', {'prediction': prediction[0]})

    # Render the form page for GET requests
    return render(request, 'form.html')
