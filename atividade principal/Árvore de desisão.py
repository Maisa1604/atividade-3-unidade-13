from sklearn.datasets import load_iris # type: ignore
from sklearn.tree import DecisionTreeClassifier, plot_tree # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Carregar dados
data = load_iris()
X, y = data.data, data.target

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Visualizar árvore
plt.figure(figsize=(12,8))
plot_tree(model, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title("Árvore de Decisão")
plt.show()

# Avaliar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy}")