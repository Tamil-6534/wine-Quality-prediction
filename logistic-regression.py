# import the required packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
# 1. Import and clean the data
data = pd.read_csv(r'winequalityN.csv')
df = data.copy()
df.dropna(axis=0, inplace=True)
df.drop(df.index[0:1460], axis=0, inplace=True)
# 2. Encode 'type' column
label = LabelEncoder()
df['type'] = label.fit_transform(df['type'])
# 3. Define features and target
# Note: We are using 4 features
features = ['volatile acidity', 'chlorides', 'density', 'alcohol']
x = df[features]
y = df['quality']
# 4. Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
# 5. Scaling
scalar = StandardScaler()
# Fit and transform the training data
x_train_scale = scalar.fit_transform(x_train)
# IMPORTANT: Use .transform() only for test data to avoid data leakage
x_test_scale = scalar.transform(x_test)

# 6. Train the model
model = LogisticRegression()
model.fit(x_train_scale, y_train)
# 7. Evaluate the model
y_pred = model.predict(x_test_scale)
print('classification_report')
print(classification_report(y_test, y_pred, zero_division=0))
# 8. Predict the user given input
# FIX: Put all 4 values into a single 2D list [[va, cl, dens, alc]]
# The order must match your 'features' list exactly
user_input_raw=[[0.66, 0.029,0.9892,15]]
#user_input_raw=[[0.12,0.01,0.989,14.5]]
# FIX: Scale the whole row at once using the already fitted scalar
input_value_scaled = scalar.transform(user_input_raw)
# FIX: Predict using the 2D scaled array
prediction = model.predict(input_value_scaled)
print(f'\nThe predicted wine quality for the given input is {prediction[0]} out of 10')
if prediction[0] >=7:
    print("RESULT: Good Quality")
else:
    print("RESULT: Bad Quality")
