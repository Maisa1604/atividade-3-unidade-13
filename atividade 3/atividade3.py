import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, roc_curve, roc_auc_score

# ==================================================
# 1. Regressão Linear (Custo de Energia Elétrica)
# ==================================================

print("\n=== Regressão Linear ===")

# Dados fictícios
X_linear = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y_linear = np.array([50, 60, 70, 80, 90])

# Treinamento do modelo
model_linear = LinearRegression()
model_linear.fit(X_linear, y_linear)

# Coeficientes
beta_0 = model_linear.intercept_
beta_1 = model_linear.coef_[0]
print(f"Coeficiente angular (β1): {beta_1}")
print(f"Intercepto (β0): {beta_0}")

# Previsões
y_pred_linear = model_linear.predict(X_linear)

# Erro Quadrático Médio (MSE)
mse = mean_squared_error(y_linear, y_pred_linear)
print(f"Erro Quadrático Médio (MSE): {mse}")

# Visualização
plt.scatter(X_linear, y_linear, color='blue', label='Dados originais')
plt.plot(X_linear, y_pred_linear, color='red', label='Reta de regressão')
plt.xlabel('Número de Aparelhos')
plt.ylabel('Custo Mensal')
plt.title('Regressão Linear')
plt.legend()
plt.show()

# ==================================================
# 2. Regressão Logística (Risco de Doença Cardíaca)
# ==================================================

print("\n=== Regressão Logística ===")

# Dados fictícios
X_logistic = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y_logistic = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

# Treinamento do modelo
model_logistic = LogisticRegression()
model_logistic.fit(X_logistic, y_logistic)

# Previsões
y_pred_logistic = model_logistic.predict(X_logistic)

# Acurácia
accuracy = accuracy_score(y_logistic, y_pred_logistic)
print(f"Acurácia: {accuracy * 100:.2f}%")

# Matriz de Confusão
cm = confusion_matrix(y_logistic, y_pred_logistic)
print("Matriz de Confusão:")
print(cm)

# Curva ROC
y_prob = model_logistic.predict_proba(X_logistic)[:, 1]
fpr, tpr, thresholds = roc_curve(y_logistic, y_prob)
auc = roc_auc_score(y_logistic, y_prob)

# Plot da curva ROC
plt.plot(fpr, tpr, label=f'Curva ROC (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Linha de Referência')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend()
plt.show()

# ==================================================
# 3. Algoritmo K-Means (Agrupamento de Alunos)
# ==================================================

print("\n=== Algoritmo K-Means ===")

# Dados fictícios
X_kmeans = np.array([[70, 75], [80, 85], [60, 65], [90, 95], [50, 55], [85, 80], [65, 70], [75, 85], [55, 60], [95, 90]])

# K-Means com 2 clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_kmeans)

# Rótulos dos clusters
print("Rótulos dos clusters:", kmeans.labels_)

# Inércia
print("Inércia:", kmeans.inertia_)

# Visualização
plt.scatter(X_kmeans[:, 0], X_kmeans[:, 1], c=kmeans.labels_, cmap='viridis', label='Dados')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroides')
plt.xlabel('Nota de Matemática')
plt.ylabel('Nota de Ciências')
plt.title('Agrupamento de Alunos com K-Means')
plt.legend()
plt.show()

# ==================================================
# 4. Árvore de Decisão (Classificação de Veículos)
# ==================================================

print("\n=== Árvore de Decisão ===")

# Dados fictícios
X_tree = np.array([[100, 1200], [150, 1500], [200, 1800], [250, 2000], [300, 2200]])
y_tree = np.array(['Econômico', 'Econômico', 'Não Econômico', 'Não Econômico', 'Não Econômico'])

# Treinamento do modelo
model_tree = DecisionTreeClassifier()
model_tree.fit(X_tree, y_tree)

# Previsões
y_pred_tree = model_tree.predict(X_tree)

# Acurácia
accuracy_tree = accuracy_score(y_tree, y_pred_tree)
print(f"Acurácia: {accuracy_tree * 100:.2f}%")

# Matriz de Confusão
cm_tree = confusion_matrix(y_tree, y_pred_tree)
print("Matriz de Confusão:")
print(cm_tree)

# Visualização da árvore
plt.figure(figsize=(10, 8))
plot_tree(model_tree, filled=True, feature_names=['Potência', 'Peso'], class_names=['Econômico', 'Não Econômico'])
plt.title('Árvore de Decisão')
plt.show()