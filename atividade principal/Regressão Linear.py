from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

# Gerar dados fictícios
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Prever
y_pred = model.predict(X_test)

# Visualizar
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.title("Regressão Linear")
plt.show()

# Avaliar
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio: {mse}")
print(f"Coeficientes: Intercepto = {model.intercept_}, Inclinação = {model.coef_}")