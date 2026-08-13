import math
from pathlib import Path

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss

from qdk import qsharp


SHOTS = 50
DEPTH = 10
ITERATIONS = 20
LEARNING_RATE = 0.15
PERTURBATION = 0.10
RANDOM_STATE = 42


def load_quantum_project():
    project_root = Path(__file__).resolve().parent
    qsharp.init(project_root=str(project_root))


def load_dataset():
    data = load_breast_cancer()

    X = data.data
    y = data.target

    return X, y


def prepare_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    scaler = MinMaxScaler(
        feature_range=(0.0, math.pi)
    )

    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )


def make_qsharp_array(values):
    return "[" + ", ".join(
        f"{float(value):.8f}"
        for value in values
    ) + "]"


def run_quantum_tree(features, angles):
    feature_string = make_qsharp_array(features)
    angle_string = make_qsharp_array(angles)

    expression = (
        f"QuantumTree.QuantumTree("
        f"{DEPTH}, "
        f"{feature_string}, "
        f"{angle_string}"
        f")"
    )

    results = qsharp.run(
        expression,
        shots=SHOTS
    )

    ones = sum(
        1
        for result in results
        if str(result) == "One"
    )

    return ones / SHOTS


def predict(X, angles):
    probabilities = []

    for sample in X:
        probability = run_quantum_tree(
            sample,
            angles
        )

        probabilities.append(probability)

    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return predictions, probabilities


def calculate_loss(y, probabilities):
    probabilities = np.clip(
        probabilities,
        1e-7,
        1 - 1e-7
    )

    return log_loss(
        y,
        probabilities,
        labels=[0, 1]
    )


def evaluate_angles(X, y, angles):
    _, probabilities = predict(
        X,
        angles
    )

    return calculate_loss(
        y,
        probabilities
    )


def train(X_train, y_train):
    rng = np.random.default_rng(
        RANDOM_STATE
    )

    angles = rng.uniform(
        -math.pi,
        math.pi,
        DEPTH
    )

    best_angles = angles.copy()

    best_loss = evaluate_angles(
        X_train,
        y_train,
        angles
    )

    print("TRAINING")
    

    print(
        f"Initial loss: {best_loss:.4f}"
    )

    for iteration in range(ITERATIONS):

        delta = rng.choice(
            [-1.0, 1.0],
            size=DEPTH
        )

        plus_angles = (
            angles
            + PERTURBATION * delta
        )

        minus_angles = (
            angles
            - PERTURBATION * delta
        )

        plus_loss = evaluate_angles(
            X_train,
            y_train,
            plus_angles
        )

        minus_loss = evaluate_angles(
            X_train,
            y_train,
            minus_angles
        )

        gradient = (
            (plus_loss - minus_loss)
            / (2.0 * PERTURBATION)
        ) * delta

        angles = (
            angles
            - LEARNING_RATE * gradient
        )

        angles = np.clip(
            angles,
            -math.pi,
            math.pi
        )

        current_loss = evaluate_angles(
            X_train,
            y_train,
            angles
        )

        if current_loss < best_loss:
            best_loss = current_loss
            best_angles = angles.copy()

        print(
            f"Iteration "
            f"{iteration + 1:2d}/{ITERATIONS} | "
            f"Loss: {current_loss:.4f} | "
            f"Best: {best_loss:.4f}"
        )

    return best_angles, best_loss


def main():

    print("HYBRID QUANTUM DECISION TREE")


    load_quantum_project()

    X, y = load_dataset()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    ) = prepare_data(X, y)

    print(
        f"\nSamples : {len(X)}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        "Classes : 0 and 1"
    )

    print(
        f"\nTree depth : {DEPTH}"
    )

    print(
        f"Tree qubits: {DEPTH}"
    )

    print(
        f"H gates    : {DEPTH}"
    )

    print(
        f"Branches   : {2 ** DEPTH}"
    )

    best_angles, best_loss = train(
        X_train,
        y_train
    )

    print("\nVALIDATION")
   

    validation_predictions, validation_probabilities = predict(
        X_validation,
        best_angles
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    validation_loss = calculate_loss(
        y_validation,
        validation_probabilities
    )

    print(
        f"Loss     : {validation_loss:.4f}"
    )

    print(
        f"Accuracy : {validation_accuracy:.4f}"
    )

    print("\nFINAL TEST")
  

    test_predictions, test_probabilities = predict(
        X_test,
        best_angles
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_loss = calculate_loss(
        y_test,
        test_probabilities
    )

    print(
        f"Loss     : {test_loss:.4f}"
    )

    print(
        f"Accuracy : {test_accuracy:.4f}"
    )

    print("\nLEARNED ANGLES")
 

    print(
        np.round(
            best_angles,
            4
        )
    )

    print("\nSAMPLE PREDICTIONS")
    

    for i in range(
        min(10, len(X_test))
    ):
        print(
            f"{i + 1:2d} | "
            f"P(1) = "
            f"{test_probabilities[i]:.3f} | "
            f"Prediction = "
            f"{test_predictions[i]} | "
            f"Actual = "
            f"{y_test[i]}"
        )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()