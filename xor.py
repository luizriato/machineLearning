import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.neural_network import MLPClassifier

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0] 

mlp = MLPClassifier(verbose=True, hidden_layer_sizes=4, max_iter=10000, tol=1e-6, activation='relu')
mlp.fit(x, y)

# Fazer previsões
print("\nPrevisões da Rede Neural:")
for caso in x:
    pred = mlp.predict([caso])[0]
    print(f"Entrada: {caso} => Saída prevista: {pred}")

# Acurácia (opcional)
acc = mlp.score(x, y)
print(f"\nAcurácia no treino: {acc:.2f}")

print("Parâmetros da Rede Neural:\n")

print(f"Classes: {mlp.classes_}")
print(f"Erro final (loss): {mlp.loss_}")
print(f"Número de amostras visitadas:", mlp.t_)
print(f"Número de atributos de entrada: ", mlp.n_features_in_)
print(f"Número de ciclos (iterações): {mlp.n_iter_}")
print(f"Número de camadas (entrada + ocultas + saída): {mlp.n_layers_}")
print(f"Tamanho das camadas ocultas: mlp.hidden_layer_sizes")
print(f"Número de neurônios na camada de saída: mlp.n_outputs")
print(f"Função de ativação: ", mlp.out_activation_)

