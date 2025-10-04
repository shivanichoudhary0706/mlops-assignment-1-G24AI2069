from sklearn.tree import DecisionTreeRegressor
from misc import load_data, preprocess, train_and_evaluate

df = load_data()
X_train, X_test, y_train, y_test = preprocess(df)

model = DecisionTreeRegressor(random_state=42)
mse = train_and_evaluate(model, X_train, X_test, y_train, y_test)
print(f'DecisionTreeRegressor Test MSE: {mse:.4f}')