import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("House_Rent_Dataset.csv")

# Clean data
df = df.drop(columns=["Posted On"], errors="ignore")

# Fix floor
df["Floor"] = df["Floor"].astype(str).str.split(" ").str[0]
df["Floor"] = pd.to_numeric(df["Floor"], errors="coerce")

df = df.dropna(subset=["Rent"])

X = df.drop("Rent", axis=1)
y = df["Rent"]

# Fill missing
for col in X.select_dtypes(include=["float64", "int64"]):
    X[col] = X[col].fillna(X[col].mean())

for col in X.select_dtypes(include=["object"]):
    X[col] = X[col].fillna(X[col].mode()[0])

# Encoding
X_encoded = pd.get_dummies(X)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_scaled, y)

# Save everything
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X_encoded.columns, open("columns.pkl", "wb"))

print("Model trained & saved ✅")