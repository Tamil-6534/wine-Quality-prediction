#ridge regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

data=pd.read_csv(r'winequalityN.csv')
data.dropna(inplace=True)
data.drop(data.index[0:2000],axis=0,inplace=True)

y = data["quality"]          # Target
X = data.drop("quality", axis=1)   # Features
X = data.drop("type", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

ridge = Ridge(alpha=35)

ridge.fit(X_train, y_train)


y_pred_ridge = ridge.predict(X_test)


ridge_mse = mean_squared_error(y_test, y_pred_ridge)

print("\nRidge Mean Squared Error:", ridge_mse)

print("\nRidge Coefficients:")
for feature, coef in zip(X.columns, ridge.coef_):
    print(feature, ":", coef)
