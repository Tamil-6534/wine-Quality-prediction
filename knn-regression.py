import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
# 1. Load and Clean
data = pd.read_csv(r'winequalityN.csv')
df = data.dropna().copy()
# 2. Encode 'type'
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])
# 3. Use all 12 features for a higher R2 score
features = ['type', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 
            'sulphates', 'alcohol']
X = df[features]
y = df['quality']
# 4. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 5. SCALING (Mandatory for KNN)
# KNN calculates distance; scaling ensures no feature "dominates" others.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 6. Train KNN Regressor
# weights='distance' makes closer neighbors more influential than far ones
model = KNeighborsRegressor(n_neighbors=9, weights='distance')
model.fit(X_train_scaled, y_train)
# 7. Predict and Evaluate
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"--- KNN Regression Performance ---")
print(f"R2 Score: {r2:.2f}")
print(f"Mean Squared Error: {mse:.4f}")
# 8. User Input Prediction
# Example: [type, fixed_ac, vol_ac, citric, sugar, chlor, free_so2, tot_so2, dens, pH, sulph, alc]
user_input = [[1, 7.0, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3.0, 0.45, 8.8]]
user_input_scaled = scaler.transform(user_input)
prediction = model.predict(user_input_scaled)
print(f"\nPredicted Wine Quality: {prediction[0]:.2f} / 10")
