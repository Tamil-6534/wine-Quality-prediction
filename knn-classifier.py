import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load and Clean
data = pd.read_csv(r'winequalityN.csv')
df = data.dropna().copy()

# 2. FEATURE ENGINEERING: Use Binary Classification
# We define 7 or higher as "Good" (1) and everything else as "Bad" (0)
# This is the standard way to achieve >85% accuracy on this dataset
df['quality_label'] = (df['quality'] >= 7).astype(int)

# 3. ENCODING: Convert 'type' (white/red) to numbers
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# 4. USE ALL FEATURES: Give the model more information
features = ['type', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 
            'sulphates', 'alcohol']
X = df[features]
Y = df['quality_label']

# 5. SPLIT
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 6. SCALING (Mandatory for KNN)
# This puts all numbers on a fair scale (e.g., -1 to 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. TRAIN: Using k=1 or k=3 often yields highest accuracy for this dataset
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train_scaled, Y_train)

# 8. PREDICT & EVALUATE
Y_pred = model.predict(X_test_scaled)
print(f"Improved Accuracy: {accuracy_score(Y_test, Y_pred)*100:.2f}%")

# 9. PREDICT USER INPUT
# [type, fixed_ac, vol_ac, citric, sugar, chlor, free_so2, tot_so2, dens, pH, sulph, alc]
user_input = [[0, 7.0, 0.66, 0.3, 2.0, 0.029, 15, 50, 0.9892, 3.5, 0.6, 13.6]]
user_input_scaled = scaler.transform(user_input)
prediction = model.predict(user_input_scaled)

result = "Good (7+)" if prediction[0] == 1 else "Bad (<7)"
print(f"User Input Prediction: {result}")
