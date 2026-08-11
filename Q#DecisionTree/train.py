import math
from pathlib import Path

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss

from qdk import qsharp


SHOTS = 50
MAX_DEPTH = 20
MAX_SIMULATION_DEPTH = 20
ITERATIONS = 5
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


def run_quantum_tree(
    features,
    angles,
    depth,
    shots=SHOTS
):
    feature_string = make_qsharp_array(features)
    angle_string = make_qsharp_array(angles)

    expression = (
        f"QuantumTree.QuantumTree("
        f"{depth}, "
        f"{feature_string}, "
        f"{angle_string}"
        f")"
    )

    results = qsharp.run(
        expression,
        shots=shots
    )

    ones = sum(
        1
        for result in results
        if str(result) == "One"
    )

    return ones / shots


def create_angles(depth, seed):
    rng = np.random.default_rng(seed)

    return rng.uniform(
        -math.pi,
        math.pi,
        depth
    )


def predict(
    X,
    depth,
    angles
):
    probabilities = []

    for sample in X:
        probability = run_quantum_tree(
            sample,
            angles,
            depth
        )

        probabilities.append(
            probability
        )

    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return predictions, probabilities


def train_depth(
    X_train,
    y_train,
    X_validation,
    y_validation,
    depth
):
    best_angles = None
    best_loss = float("inf")

    for iteration in range(ITERATIONS):

        angles = create_angles(
            depth,
            RANDOM_STATE + depth * 100 + iteration
        )

        _, probabilities = predict(
            X_train,
            depth,
            angles
        )

        loss = log_loss(
            y_train,
            probabilities,
            labels=[0, 1]
        )

        if loss < best_loss:
            best_loss = loss
            best_angles = angles.copy()

    predictions, probabilities = predict(
        X_validation,
        depth,
        best_angles
    )

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    return (
        accuracy,
        best_loss,
        best_angles
    )


def search_depths(
    X_train,
    y_train,
    X_validation,
    y_validation
):
    best_depth = None
    best_accuracy = -1.0
    best_angles = None

    results = []

    for depth in range(1, MAX_DEPTH + 1):

        theoretical_branches = 2 ** depth

        print(
            f"Depth {depth:4d} | "
            f"Qubits {depth:4d} | "
            f"Branches 2^{depth}"
        )

        if depth > MAX_SIMULATION_DEPTH:
            print(
                "Simulation limit reached."
            )
            break

        accuracy, loss, angles = train_depth(
            X_train,
            y_train,
            X_validation,
            y_validation,
            depth
        )

        results.append(
            (
                depth,
                accuracy,
                loss
            )
        )

        print(
            f"Validation accuracy: "
            f"{accuracy:.4f}"
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_depth = depth
            best_angles = angles.copy()

    return (
        best_depth,
        best_accuracy,
        best_angles,
        results
    )


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

  
    print(f"Samples : {len(X)}")
    print(f"Features: {X.shape[1]}")
    print("Classes : 0 and 1")

    print("Searching tree depths...")
  

    (
        best_depth,
        best_accuracy,
        best_angles,
        results
    ) = search_depths(
        X_train,
        y_train,
        X_validation,
        y_validation
    )

   
    print("BEST QUANTUM TREE")


    print(
        f"Depth: {best_depth}"
    )

    print(
        f"Qubits: {best_depth}"
    )

    print(
        f"H gates: {best_depth}"
    )

    print(
        f"Theoretical branches: 2^{best_depth}"
    )

    print(
        f"Validation accuracy: "
        f"{best_accuracy:.4f}"
    )

    print("FINAL TEST")


    predictions, probabilities = predict(
        X_test,
        best_depth,
        best_angles
    )

    test_accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy:.4f}"
    )

    print()
    print("Sample predictions:")

    for i in range(
        min(10, len(X_test))
    ):
        print(
            f"{i + 1:2d} | "
            f"P(1) = {probabilities[i]:.3f} | "
            f"Prediction = {predictions[i]} | "
            f"Actual = {y_test[i]}"
        )


if __name__ == "__main__":
    main()