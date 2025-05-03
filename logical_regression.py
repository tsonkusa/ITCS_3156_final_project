# --- Logistic Regression for Stock Movement Prediction ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Honda_Data.csv')  # Replace with your actual file path

# Preprocess
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)

# Target: 1 if next day's Close is higher
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Features and labels
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
y = df['Target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = model.score(X_test, y_test)
print(f"\n📊 Logistic Regression Accuracy: {accuracy:.2f}")
print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))
print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize Predictions vs Actual
plt.figure(figsize=(12, 4))
plt.plot(y_pred[:100], label='Predicted')
plt.plot(y_test.values[:100], label='Actual', alpha=0.7)
plt.legend()
plt.title('Logistic Regression: Predictions vs Actual (First 100)')
plt.show()
