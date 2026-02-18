import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
data = pd.read_csv("data/motor_data.csv")

# Clean column names
data.columns = data.columns.str.strip()

# Target
target = "pm"

# Use ONLY 7 features
features = [
    "ambient",
    "coolant",
    "u_d",
    "u_q",
    "motor_speed",
    "i_d",
    "i_q"
]

X = data[features]
y = data[target]

print("Features used:", X.columns.tolist())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = DecisionTreeRegressor(max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Save model + feature order
joblib.dump((model, features), "model/motor_temp_model.pkl")

print("Model trained and saved!")
