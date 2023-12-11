import os
import joblib
import pandas as pd
import numpy as np

# Load the models
models = ["XGBoost", "RandomForest", "AdaBoost"]
classifiers = {model: joblib.load(f'saved_models/{model}_model.pkl') for model in models}

# Load the test data
test_data = pd.read_csv('cleaned_data/features_train.csv', index_col=0)

# Load the target data
target_train = pd.read_csv('cleaned_data/target_train.csv', index_col=0)

def interactive_probe_with_models(amount):
    print("Simulating a payment with amount:", amount)

    # Select 5 random paths from the test data that match the user's input amount
    selected_data = test_data[test_data['path_amount'] == amount].sample(n=5)

    for i, path in selected_data.iterrows():
        print("\nSelected path:", path)

        # Make predictions with each model for the current path
        predictions = {model: clf.predict([path]) for model, clf in classifiers.items()}

        # Print the predictions
        for model, pred in predictions.items():
            print(f"Prediction by {model}:", pred)

        # Check if at least two models agree
        if sum(pred[0] for pred in predictions.values()) >= 2:
            print("At least two models agree for this path. Proceeding with this path.")
            
            # Check if the predictions match the target_train
            actual = target_train.loc[i]
            for model, pred in predictions.items():
                print(f"Does the prediction by {model} match the target_train? {'Yes' if pred[0] == actual else 'No'}")

            return path

    print("No path found where at least two models agree.")
    return None


if __name__ == "__main__":
    amount = input("Please enter an amount: ")
    interactive_probe_with_models(amount)
