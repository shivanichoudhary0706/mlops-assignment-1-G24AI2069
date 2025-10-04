from sklearn.kernel_ridge import KernelRidge
from misc import load_data, preprocess, train_and_evaluate

df = load_data()
X_train, X_test, y_train, y_test = preprocess(df)

model = KernelRidge(alpha=1.0)
mse = train_and_evaluate(model, X_train, X_test, y_train, y_test)
print(f'KernelRidge Test MSE: {mse:.4f}')