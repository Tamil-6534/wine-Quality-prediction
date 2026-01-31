#lasso regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

data=pd.read_csv(r'winequalityN.csv')
data.dropna(inplace=True)

y = data["quality"]          # Target
X = data.drop(["quality", "type"], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

lasso = Lasso(alpha=0.1)

lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

lasso_mse = mean_squared_error(y_test, y_pred_lasso)

print("\nLasso Mean Squared Error:", lasso_mse)

print("\nLasso Coefficients:")
for feature, coef in zip(X.columns, lasso.coef_):
    print(feature, ":", coef)



