"""
train_model.py
===============
This script TRAINS a machine learning model that can look at a flower's
measurements (sepal length/width, petal length/width) and guess which of
the 3 Iris species it is: setosa, versicolor, or virginica.

You only need to run this ONCE (or whenever you want to retrain).
It saves the trained model to a file called "model.pkl", which the API
(api.py) will load later.

Run it like this:
    python train_model.py

New to ML? Here's the flow in plain English:
1. Load some example flower data where we already know the species.
2. Split it into a "training" set (to teach the model) and a "test" set
   (to check how well it learned).
3. Train a RandomForestClassifier -- think of it as a group of many small
   decision trees that vote together on the answer.
4. Check its accuracy on the test set.
5. Save the trained model to disk so other programs can reuse it.
"""

import joblib  # joblib saves/loads Python objects (like our trained model) to/from a file
from sklearn.datasets import load_iris  # a small built-in dataset of flower measurements
from sklearn.ensemble import RandomForestClassifier  # the ML algorithm we'll use
from sklearn.model_selection import train_test_split  # splits data into train/test parts
from sklearn.metrics import accuracy_score  # measures how many predictions were correct


def main():
    # ---- Step 1: Load the data ----
    # `data.data` is a table of 4 numbers per flower (the measurements).
    # `data.target` is the correct answer (0, 1, or 2) for each flower.
    # `data.target_names` translates 0/1/2 into ["setosa", "versicolor", "virginica"].
    data = load_iris()
    X = data.data                       # X = the input features (measurements)
    y = data.target                     # y = the correct labels (species, as numbers)
    target_names = list(data.target_names)

    # ---- Step 2: Split into training and test sets ----
    # We hold back 20% of the flowers as a "test" the model never sees while
    # learning. This tells us how well it generalizes to new, unseen flowers.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,      # 20% for testing, 80% for training
        random_state=42,    # a fixed "seed" so results are reproducible
        stratify=y,         # keep the same species proportions in both splits
    )

    # ---- Step 3: Train the model ----
    # n_estimators=200 means the forest is made of 200 decision trees.
    # More trees can improve accuracy but take a bit longer to train.
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)  # this is where the actual "learning" happens

    # ---- Step 4: Evaluate the model ----
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy:.2%}")  # e.g. "Test accuracy: 90.00%"

    # ---- Step 5: Save the model to disk ----
    # We bundle the model together with target_names so the API doesn't need
    # to know about scikit-learn's internal numbering (0, 1, 2) -- it can just
    # look up the real species name.
    joblib.dump({"model": model, "target_names": target_names}, "model.pkl")
    print("Saved trained model to model.pkl")


# This line means "only run main() if this file is executed directly"
# (as opposed to being imported by another file).
if __name__ == "__main__":
    main()
