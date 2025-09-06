from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 
  
# metadata 
print(predict_students_dropout_and_academic_success.metadata) 
  
# variable information 
print(predict_students_dropout_and_academic_success.variables) 

#%% CARGA DOS DADOS
# ID 697 -> Predict Students' Dropout and Academic Success
dataset = fetch_ucirepo(id=697)

X = dataset.data.features
y = dataset.data.targets

print("Shape X:", X.shape)
print("Shape y:", y.shape)

#%% PRÉ-PROCESSAMENTO
# Se houver variáveis categóricas, aplicar LabelEncoder
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Alvo (y) também pode precisar de encoding
if y.select_dtypes(include=["object"]).shape[1] > 0:
    y = y.apply(LabelEncoder().fit_transform)

# Converter y para Series (se for DataFrame com 1 coluna)
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

#%% DIVISÃO TREINO/TESTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Escalonamento
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% CLASSIFICAÇÃO COM REDE NEURAL (MLP)
clf = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)
clf.fit(X_train, y_train)

#%% AVALIAÇÃO
y_pred = clf.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))