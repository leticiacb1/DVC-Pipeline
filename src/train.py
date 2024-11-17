# Data
import pandas as pd

# Export
import pickle

# Plot
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder


def load_data():
    df = pd.read_parquet("data/bank_preproc.parquet")
    return df


def split(df):
    X = df.drop("deposit", axis=1)
    y = df["deposit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )

    return X_train, X_test, y_train, y_test


def train_ohe(X_train):
    cat_cols = ["job", "marital", "education", "housing"]
    one_hot_enc = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    one_hot_enc.fit(X_train)

    return one_hot_enc


def train(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50, max_depth=5)
    model.fit(X_train, y_train)
    return model


def export_model(model, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(model, f)


def export_results(model, X_test, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    # Create a DataFrame with the evaluation metrics
    results_df = pd.DataFrame(
        {"Accuracy": [accuracy], "Precision": [precision], "Recall": [recall]}
    )

    results_df.to_csv("results/model_test_metrics.csv", index=False)


def export_confusion_matrix(model, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    # Create a pandas DataFrame for the confusion matrix
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)

    # Generate the confusion matrix plot
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig("results/confusion_matrix.png")
    plt.close()


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    ohe = train_ohe(X_train)
    X_train = ohe.transform(X_train)
    X_test = ohe.transform(X_test)

    model = train(X_train, y_train)
    y_pred = model.predict(X_test)

    export_results(model, X_test, y_test, y_pred)
    export_confusion_matrix(model, y_test, y_pred)
    export_model(ohe, "models/ohe.pickle")
    export_model(model, "models/model.pickle")


if __name__ == "__main__":
    main()