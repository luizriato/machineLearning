from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#%% CARGA DOS DADOS
dataset = fetch_ucirepo(id=697)

X = dataset.data.features
y = dataset.data.targets

# Pré-processamento
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

if y.select_dtypes(include=["object"]).shape[1] > 0:
    y = y.apply(LabelEncoder().fit_transform)

if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Escalonamento
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% DEFINIÇÃO DE ARQUITETURAS
arquiteturas = [
    (20,),   # 1 camada, 20 neurônios
    (50,),   # 1 camada, 50 neurônios
    (100,),  # 1 camada, 100 neurônios
    (20,20),   # 2 camadas: 20 e 20 neurônios
    (50,20),   # 2 camadas: 50 e 20 neurônios
    (100,20),  # 2 camadas: 100 e 20 neurônios
    (100,50)   # 2 camadas: 100 e 50 neurônios
]

resultados = []

for arch in arquiteturas:
    clf = MLPClassifier(hidden_layer_sizes=arch, max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    resultados.append({
        "Arquitetura": arch,
        "Acurácia": acc,
        "Matriz de Confusão": cm
    })

#%% RELATÓRIO
print("===== RELATÓRIO DE RESULTADOS =====\n")
for r in resultados:
    print(f"Arquitetura: {r['Arquitetura']}")
    print(f"Acurácia: {r['Acurácia']:.4f}")
    print("Matriz de Confusão:")
    print(r["Matriz de Confusão"])
    print("-"*50)

    # Criar DataFrame com os resultados
tabela_resultados = pd.DataFrame([
    {"Arquitetura": r["Arquitetura"], "Acurácia": r["Acurácia"]}
    for r in resultados
])

print("\n===== TABELA DE RESULTADOS =====\n")
print(tabela_resultados.to_string(index=False))
