import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# --- Load Data ---
try:
    data = pd.read_csv("final_salary_data.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'final_salary_data.csv' not found.")
    exit()

# --- Preprocessing ---
target_column = 'Salary_USD'
features_data = data.drop(columns=[target_column])
target_data = data[target_column]

if 'Location' in features_data.columns:
    features_data['Location'] = features_data['Location'].astype(str)
    mode_location = features_data['Location'].mode()[0]
    features_data['Location'].replace('?', mode_location, inplace=True)
    print(f"Replaced '?' in Location with {mode_location}")

categorical_cols = features_data.select_dtypes(include='object').columns.tolist()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    features_data[col] = le.fit_transform(features_data[col])
    label_encoders[col] = le

print("Categorical encoding done.")

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(features_data, target_data, test_size=0.2, random_state=42)

# --- Model Pipeline ---
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

print("Training model...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# --- Evaluation ---
y_pred = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

# --- Save Artifacts ---
joblib.dump(model_pipeline, "best_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(features_data.columns.tolist(), "feature_columns.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(y_pred, "y_pred.pkl")
joblib.dump(mae, "mae_score.pkl")
joblib.dump(r2, "r2_score.pkl")
joblib.dump(rmse, "rmse_score.pkl")

print("Artifacts saved.")