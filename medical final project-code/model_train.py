import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Uploading dataset
data = pd.read_csv("uploads/Heart_Disease_Prediction.csv")

# change target row ("Presence" -> 1, "Absence" -> 0)
data["Heart Disease"] = data["Heart Disease"].map({"Presence": 1, "Absence": 0})

# Reducing weight to manipulate blood pressure
data["BP"] = data["BP"] / data["BP"].max()  # Normalleştirme işlemi

# features and target variables
X = data.drop(["Heart Disease", "index"], axis=1)  # Hedef sütunu ve index sütununu kaldırın
y = data["Heart Disease"]

# seperate data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# training model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# accuracy
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Print feature importances
importances = model.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance}")

# save model
joblib.dump(model, "heart_disease_model.pkl")
