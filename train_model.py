
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("Crop_recommendation.csv")

# Separate input features and target
X = data[["N", "P", "K", "temperature",
          "humidity", "ph", "rainfall"]]
y = data["label"]

# Train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

# Save the trained model
joblib.dump(model, "model.joblib")

print("Model trained and saved successfully!")
