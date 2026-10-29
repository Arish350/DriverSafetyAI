
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



# Load dataset
data = pd.read_csv("drowsiness_data.csv")

data = data.dropna(subset=["ear", "mouth_ratio", "label"])
data = data[data["label"].isin(["alert", "drowsy"])]

print("Dataset counts:")
print(data["label"].value_counts())

if data["label"].nunique() < 2:
    raise ValueError("Both alert and drowsy samples are required.")

X = data[["ear", "mouth_ratio"]]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42
    )
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(model, "drowsiness_model.pkl")

print("\nSuccess! drowsiness_model.pkl created.")
