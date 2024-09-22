# Importación de librerías
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter  # Para verificar la distribución de clases

# Cargar los datos
df = pd.read_csv("financiera_datos_simulados.csv")

# Preprocesamiento de los datos
# Supongamos que las características están en todas las columnas excepto la última (etiqueta)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balancear los datos de entrenamiento con SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# Verificar la distribución de clases después de SMOTE
print(f"Distribución de clases después de SMOTE: {Counter(y_resampled)}")

# Modelo de Random Forest con GridSearchCV para ajustar los hiperparámetros
rf_model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'max_features': ['sqrt', 'sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Reducir el número de splits en la validación cruzada
cv = StratifiedKFold(n_splits=3)  # Cambiado de 5 a 3

grid_search = GridSearchCV(rf_model, param_grid, cv=cv, scoring='accuracy')
grid_search.fit(X_resampled, y_resampled)

# Mejor modelo de Random Forest
best_rf_model = grid_search.best_estimator_

# Predicciones y evaluación del modelo de Random Forest
y_pred_rf = best_rf_model.predict(X_test_scaled)

print("Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Best Parameters (Random Forest):", grid_search.best_params_)
print("Best Accuracy (Random Forest):", grid_search.best_score_)

# Modelo de Regresión Logística
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_resampled, y_resampled)

# Predicciones y evaluación del modelo de Regresión Logística
y_pred_log = log_reg_model.predict(X_test_scaled)

print("\nLogistic Regression Results:")
print("Accuracy (Logistic Regression):", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix (Logistic Regression):\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred_log))

# Comparación de modelos
print("\nComparación de Modelos:")
print(f"Mejor precisión del modelo Random Forest: {grid_search.best_score_}")
print(f"Precisión del modelo de Regresión Logística: {accuracy_score(y_test, y_pred_log)}")
