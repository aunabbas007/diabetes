import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    # Replace invalid zeros with NaN
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

    # Fill missing values with median
    df[cols_with_zero] = df[cols_with_zero].fillna(df[cols_with_zero].median())

    return df


def split_data(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def find_best_k(X_train, X_test, y_train, y_test):
    accuracies = []

    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    best_k = accuracies.index(max(accuracies)) + 1
    return best_k, accuracies


def train_final_model(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    df = load_data("data/diabetes.csv")

    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train, X_test = scale_data(X_train, X_test)

    best_k, accuracies = find_best_k(X_train, X_test, y_train, y_test)
    print(f"Best K: {best_k}")

    model = train_final_model(X_train, y_train, best_k)

    evaluate_model(model, X_test, y_test)