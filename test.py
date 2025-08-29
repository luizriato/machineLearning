import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

data = {
    'Tempo': ['Ensolarado', 'Ensolarado', 'Nublado', 'Chuvoso', 'Chuvoso', 'Chuvoso',
              'Nublado', 'Ensolarado', 'Ensolarado', 'Chuvoso', 'Ensolarado', 'Nublado',
              'Nublado', 'Chuvoso'],
    'Temperatura': ['Quente', 'Quente', 'Quente', 'Ameno', 'Frio', 'Frio',
                    'Frio', 'Ameno', 'Frio', 'Ameno', 'Ameno', 'Ameno',
                    'Quente', 'Ameno'],
    'Umidade': ['Alta', 'Alta', 'Alta', 'Alta', 'Normal', 'Normal',
                'Normal', 'Alta', 'Normal', 'Normal', 'Normal', 'Alta',
                'Normal', 'Alta'],
    'Vento': ['Fraco', 'Forte', 'Fraco', 'Fraco', 'Fraco', 'Forte',
              'Forte', 'Fraco', 'Fraco', 'Fraco', 'Forte', 'Forte',
              'Fraco', 'Forte'],
    'Jogar': ['Não', 'Não', 'Sim', 'Sim', 'Sim', 'Não',
              'Sim', 'Não', 'Sim', 'Sim', 'Sim', 'Sim',
              'Sim', 'Não']
}

# Criar DataFrame
df = pd.DataFrame(data)

le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

# Separar variáveis independentes (X) e dependente (y)
X = df.drop('Jogar', axis=1)
y = df['Jogar']

# Treinar modelo de árvore de decisão
model = DecisionTreeClassifier()
model.fit(X, y)

# Mostrar árvore de decisão em formato de texto
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)