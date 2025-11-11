import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.pkl")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Encode categorical columns
for col in ["gender", "smoking", "alcohol", "physical_activity", "disease"]:
    df[col] = LabelEncoder().fit_transform(df[col])

# Features & target
X = df.drop("disease", axis=1)
y = df["disease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"✅ Model trained with accuracy: {model.score(X_test, y_test):.2f}")

# Save model
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
