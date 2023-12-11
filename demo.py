import os
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def pretty_print_path(path):
    hops = []
    for i in range(13):  # Assuming there are 13 hops
        if path[f'hop_{i}_destination_pubkey'] != -1.0:
            hops.append(f"Hop_{i}")
    return ' -> '.join(hops)

# Load the machine learning models
models = ["XGBoost", "RandomForest", "AdaBoost"]
classifiers = {model: joblib.load(f'saved_models/{model}_model.pkl') for model in models}
print("Loaded models:", ", ".join(models))

# Load the test data
print("Loading test data...")
features_test = pd.read_csv('cleaned_data/features_test.csv', index_col=0)

# Load the target data
print("Loading target data...")
target_test = pd.read_csv('cleaned_data/target_test.csv', index_col=0)

def interactive_probe_with_models(amount):
    print("\nSimulating a payment with amount:", amount)
    print("Finding the closest paths based on amount...")

    # Calculate the absolute difference between 'path_amount' and the input amount
    features_test['diff'] = abs(features_test['path_amount'] - amount)

    # Sort the DataFrame by the difference
    sorted_data = features_test.sort_values('diff')

    for batch_start in range(0, len(sorted_data), 5):
        # Select the next 5 rows
        selected_data = sorted_data.iloc[batch_start:batch_start+5]
        print("Selected paths based on the amount (path indices):", selected_data.index.tolist())

        for i, path in selected_data.iterrows():
            print(f"\nEvaluating path: {i}")
            # Make predictions with each model for the current path
            predictions = {model: clf.predict(np.array(path.drop('diff')).reshape(1, -1)) for model, clf in classifiers.items()}
            # Print the predictions
            for model, pred in predictions.items():
                print(f"Prediction by {model} (success=1, fail=0):", pred[0])

            # Check if at least two models agree
            agreement_count = sum(pred[0] for pred in predictions.values())
            if agreement_count >= 2:
                print(f"{agreement_count} out of 3 models agree for this path. Likely to succeed.")
                
                # Pretty print the path
                print("Selected Path:", i)
                print(pretty_print_path(path))
                
                # Check if the predictions match the target_train
                if i in target_test.index:
                    actual = target_test.loc[i, 'path_failure']
                    assert np.isscalar(actual), f"Expected a scalar value at index {i}, but got {actual}"
                else:
                    print(f"Index {i} not found in target_train.")
                    continue
                for model, pred in predictions.items():
                    match = 'Yes' if pred[0] == actual else 'No'
                    print(f"Does the prediction by {model} match the target_train? {match}")

                return i

    print("No path found where at least two models agree.")
    return None

if __name__ == "__main__":
    try:
        amount = float(input("Please enter a payment amount: "))
        selected_path = interactive_probe_with_models(amount)
        if selected_path is not None:
            print("\nSelected Path for Payment:", selected_path)
        else:
            print("No suitable path found for the entered amount.")
    except ValueError:
        print("Invalid amount entered. Please enter a numeric value.")
