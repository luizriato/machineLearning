from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier

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
# 2) Treinar Rede Neural
# =========================
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
clf.fit(X_train, y_train)

# =========================
# 3) Previsão para um aluno
# =========================
# Criamos um dicionário com todos os 36 atributos
novo_aluno_dict = {
    "Marital Status": 1,                          # 1 = solteiro
    "Application mode": 1,                        # exemplo
    "Application order": 1,
    "Course": 9085,                               # Enfermagem
    "Daytime/evening attendance": 0,              # 0 = noturno
    "Previous qualification": 1,                  # Ensino secundário
    "Previous qualification (grade)": 12.0,       # nota no secundário (exemplo)
    "Nacionality": 1,                             # 1 = Português
    "Mother's qualification": 29,                 # 9º ano
    "Father's qualification": 25,                 # Curso complementar
    "Mother's occupation": 0,
    "Father's occupation": 0,
    "Admission grade": 100,
    "Displaced": 0,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 1,                                  # 1 = masculino (exemplo)
    "Scholarship holder": 0,
    "Age at enrollment": 39,
    "International": 0,
    # Dados do 1º semestre
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 5,
    "Curricular units 1st sem (evaluations)": 5,
    "Curricular units 1st sem (approved)": 5,
    "Curricular units 1st sem (grade)": 13,
    "Curricular units 1st sem (without evaluations)": 0,
    # Dados do 2º semestre
    "Curricular units 2nd sem (credited)": 0,
    "Curricular units 2nd sem (enrolled)": 5,
    "Curricular units 2nd sem (evaluations)": 5,
    "Curricular units 2nd sem (approved)": 5,
    "Curricular units 2nd sem (grade)": 14,
    "Curricular units 2nd sem (without evaluations)": 0,
    # Indicadores macroeconômicos
    "Unemployment rate": 8.0,
    "Inflation rate": 1.5,
    "GDP": 2.0,
}


# Criar DataFrame com os mesmos atributos do dataset
novo_aluno = pd.DataFrame([novo_aluno_dict], columns=X.columns)

# Garantir encoding igual ao treino (se fosse string)
for col in novo_aluno.columns:
    if col in label_encoders:
        novo_aluno[col] = label_encoders[col].transform(novo_aluno[col])

# Escalar
novo_aluno_scaled = scaler.transform(novo_aluno)

# Previsão
pred = clf.predict(novo_aluno_scaled)[0]

# Mapear target de volta para rótulo original
y_encoder = LabelEncoder()
y_encoder.fit(dataset.data.targets.values.ravel())

print("Classe prevista (código):", pred)
print("Classe prevista (rótulo):", y_encoder.inverse_transform([pred])[0])

