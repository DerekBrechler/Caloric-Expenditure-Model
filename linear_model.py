import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Load and prepare data
import os
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "physiologically_grounded_dataset_corrected.csv"))


# Features and target
features = ["gender", "age", "weight_kg", "duration_min", "time_below_lthr", "time_above_lthr"]
X = df[features]
y = df["calories"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Results
print(f"MAE: {mae:.2f} kcal")
print(f"R²: {r2:.4f}")

# Feature coefficients
print("\nFeature Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"  {feature:20s}: {coef:8.2f}")
print(f"  {'Intercept':20s}: {model.intercept_:8.2f}")
