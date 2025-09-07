# CSC510 Foundations of Artificial Intelligence
# Module 6: Naive Bayes Classifier Iris Dataset

# Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer


# Main function
def main():

# Load iris dataset sklearn
    iris = load_iris(as_frame=True)
    X = iris.data.to_numpy()
    y = iris.target
    target_names = iris.target_names
    feature_names = iris.feature_names
    df = iris.frame


# Preview iris dataset via pandas
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df["species"] = [iris.target_names[i] for i in iris.target]
    print("Edgar Anderson Iris Dataset")
    print("Iris Classes:", target_names)
    print("Preview Iris dataset:")
    print(iris_df.head(), "\n")

# Frequency & Likelihood Tables
# Continuous features binned for tabulation
    n_bins = 3
    kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    X_binned = kb.fit_transform(X).astype(int)

    df_binned = pd.DataFrame(X_binned, columns=feature_names)
    df_binned["species"] = y

# Frequency tables: counts of bins per class
    for col in feature_names:
        freq = pd.crosstab(df_binned[col], df_binned["species"], rownames=[f"{col}_bin"], colnames=["class"])
        print(f"\nFeature: {col}")
        print(freq)

# Likelihood tables: Laplace-smoothed probabilities
    print("\n=== Likelihood Tables (Laplace-corrected) ===")
    for j, col in enumerate(feature_names):
        K = n_bins
        print(f"\nFeature: {col}")
        for cls in np.unique(y):
            mask = (y == cls)
            counts = np.bincount(X_binned[mask, j], minlength=K)
            smoothed = (counts + 1.0) / (counts.sum() + K)  # Laplace smoothing
            pretty = ", ".join([f"bin{b}={p:.3f}" for b, p in enumerate(smoothed)])
            print(f"  Class={target_names[cls]} -> {pretty}")

# Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

# Train Gaussian Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("Model trained successfully.\n")

# Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(f"Accuracy: {accuracy:.4f}\n")

# Demo: Random flower measurements
    print("=== Demo: Random Flower Measurements ===")

# Reasonable measurement ranges (based on iris dataset stats)
    sepal_length = round(random.uniform(4.0, 8.0), 2)
    sepal_width  = round(random.uniform(2.0, 4.5), 2)
    petal_length = round(random.uniform(1.0, 7.0), 2)
    petal_width  = round(random.uniform(0.1, 2.5), 2)

    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_class = model.predict(sample)[0]
    pred_proba = model.predict_proba(sample)[0]

    print(f"Generated Sample: {sample.tolist()[0]}")
    print("Posterior Probabilities:")
    for idx, cls in enumerate(target_names):
        print(f"P(Class={cls} | sample) = {pred_proba[idx]:.4f}")

    print("Final Prediction:", target_names[pred_class])

# Visualization
    print("\nGenerating visualization...")

# Plot only petal length vs petal width
    plt.figure(figsize=(8,6))
    colors = ["red", "green", "blue"]

    for i, cls in enumerate(np.unique(y)):
        plt.scatter(
            X[y == cls, 2],  # Petal length
            X[y == cls, 3],  # Petal width
            c=colors[i],
            label=target_names[cls],
            alpha=0.6
        )

# Plot the generated random point
    plt.scatter(
        sample[0, 2], sample[0, 3],
        c="black",
        marker="*",
        s=200,
        edgecolors="yellow",
        linewidths=1.5,
        label="Random Sample"
    )

    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title("Iris Dataset with Random Sample Overlay")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
