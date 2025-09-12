from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# 1) Carregar dataset
# =========================
dataset = fetch_ucirepo(id=697)
X = dataset.data.features.copy()
y = dataset.data.targets.copy()

# Encoding de variáveis categóricas
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encoding do target (caso seja string)
if y.select_dtypes(include=["object"]).shape[1] > 0:
    y = y.apply(LabelEncoder().fit_transform)

# Converter y para Series (se tiver mais de uma coluna)
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Normalização
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 2) Testar diferentes arquiteturas
# =========================
arquiteturas = [
    (20,), (50,), (100,),     # 1 camada
    (20, 20), (50, 20), (100, 20), (100, 50)  # 2 camadas
]

resultados = []

for arch in arquiteturas:
    clf = MLPClassifier(hidden_layer_sizes=arch, max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    resultados.append({
        "Arquitetura": arch,
        "Acurácia": acc,
        "Matriz de Confusão": cm
    })

# =========================
# 3) RELATÓRIO
# =========================
print("\n===== RELATÓRIO DE RESULTADOS =====\n")
for r in resultados:
    print(f"Arquitetura: {r['Arquitetura']}")
    print(f"Acurácia: {r['Acurácia']:.4f}")
    print("Matriz de Confusão:")
    print(r["Matriz de Confusão"])
    print("-"*50)

# =========================
# 4) Previsão para novo aluno
# =========================
novo_aluno_dict = {
    "Marital Status": 1,
    "Application mode": 1,
    "Application order": 1,
    "Course": 9085,
    "Daytime/evening attendance": 0,
    "Previous qualification": 1,
    "Previous qualification (grade)": 12.0,
    "Nacionality": 1,
    "Mother's qualification": 29,
    "Father's qualification": 25,
    "Mother's occupation": 0,
    "Father's occupation": 0,
    "Admission grade": 100,
    "Displaced": 0,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 1,
    "Scholarship holder": 0,
    "Age at enrollment": 39,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 5,
    "Curricular units 1st sem (evaluations)": 5,
    "Curricular units 1st sem (approved)": 5,
    "Curricular units 1st sem (grade)": 13,
    "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0,
    "Curricular units 2nd sem (enrolled)": 5,
    "Curricular units 2nd sem (evaluations)": 5,
    "Curricular units 2nd sem (approved)": 5,
    "Curricular units 2nd sem (grade)": 14,
    "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 8.0,
    "Inflation rate": 1.5,
    "GDP": 2.0,
}

novo_aluno = pd.DataFrame([novo_aluno_dict], columns=X.columns)

# Aplicar os mesmos encoders (se fosse string)
for col in novo_aluno.columns:
    if col in label_encoders:
        novo_aluno[col] = label_encoders[col].transform(novo_aluno[col])

# Escalar
novo_aluno_scaled = scaler.transform(novo_aluno)

# Usar o melhor modelo (último treinado, por ex: (100,50))
melhor_modelo = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
melhor_modelo.fit(X_train, y_train)
pred = melhor_modelo.predict(novo_aluno_scaled)[0]

# Reconstruir labels do target
y_encoder = LabelEncoder()
y_encoder.fit(dataset.data.targets.values.ravel())

print("\n===== PREVISÃO PARA NOVO ALUNO =====")
print("Classe prevista (código):", pred)
print("Classe prevista (rótulo):", y_encoder.inverse_transform([pred])[0])


# Criar DataFrame com os resultados
tabela_resultados = pd.DataFrame([
    {"Arquitetura": r["Arquitetura"], "Acurácia": r["Acurácia"]}
    for r in resultados
])

print("\n===== TABELA DE RESULTADOS =====\n")
print(tabela_resultados.to_string(index=False))