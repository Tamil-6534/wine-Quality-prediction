import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
# 1. Load and Clean
data = pd.read_csv(r'F:\internship\winequalityN.csv')
df = data.dropna().copy()
# 2. Encode categorical data
label = LabelEncoder()
df['type'] = label.fit_transform(df['type'])

# 3. Use ALL features to provide maximum information
features = ['type', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 
            'sulphates', 'alcohol']
x = df[features]
y = df['quality']
# 4. STEP ONE: Generate Polynomial Features (Degree 2)
# This creates interaction terms like alcohol * volatile acidity
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)
# 5. Split the NEW polynomial data
x_train, x_test, y_train, y_test = train_test_split(x_poly, y, train_size=0.8, random_state=42)
# 6. STEP TWO: Scaling (Crucial for Polynomial Regression)
scalar = StandardScaler()
x_train_scale = scalar.fit_transform(x_train)
x_test_scale = scalar.transform(x_test)

# 7. Train Linear Regression on the curved data
model = LinearRegression()
model.fit(x_train_scale, y_train)
# 8. Evaluation
y_pred = model.predict(x_test_scale)
r2score = r2_score(y_test, y_pred)
mse=mean_squared_error(y_test,y_pred)
# 1. Your raw input (12 features)
user_input_raw = [[
    1,      # type (White)
    6.8,    # fixed acidity
    0.10,   # VOLATILE ACIDITY (Minimum)
    0.45,   # CITRIC ACID (High/Fresh)
    1.1,    # residual sugar
    0.009,  # CHLORIDES (Minimum)
    35.0,   # free sulfur dioxide
    115.0,  # total sulfur dioxide
    0.988,  # DENSITY (Minimum)
    3.25,   # pH
    0.85,   # SULPHATES
    14.9    # ALCOHOL (Maximum)
]]
# 2. Transform to Polynomial (Turns 12 columns into 90)
input_poly = poly.transform(user_input_raw)
# 3. Scale the data (This stops the "exploding" -93.36 result)
input_scaled = scalar.transform(input_poly)
# 4. Predict
raw_prediction = model.predict(input_scaled)[0]
# 5. Manual "Clip" using Python's built-in functions (No Numpy)
# This ensures the score is at least 0 and at most 10
final_score = max(0, min(10, raw_prediction))
print(f'Predicted Wine Quality: {final_score:.2f}')
print(f"--- High Performance Linear Regression ---")
print(f'Enhanced R2 Score: {r2score:.2f}')
print(f'mse: {mse}')
