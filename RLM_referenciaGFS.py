# -*- coding: utf-8 -*-
"""
RLM para datos de cucao
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #graficos
import seaborn as sns
# rlm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
# ---------------------------------------------------------------------------------------------
# archivo .txt con los datos de Cucao
file_path = r'C:\Users\fzura\Desktop\2024-2\HP\DATOS_6var\cucao_temp_min_max_avg_202101.txt'

# skiprows -> para evitar encabezados no deseados
CUCAO = pd.read_csv(file_path, sep=r'\s+', encoding='latin1', skiprows=2, 
                    names=["Dia", "Temp_Mean", "Temp_Max", "Temp_Min", "Pres", "U_Wind", "V_Wind"])

# verifico datos
print(CUCAO.head())

# matriz de correlación
# --------------------------------------------------------------------------------------------
# selecciono variables que se correlacionaran
corr_matrix = CUCAO[['Temp_Max', 'Pres', 'U_Wind', 'V_Wind']].corr()

# gradico matriz
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación entre las Variables Predictoras y la Temperatura Máxima')
plt.show()

# ahora veo las variables predictoras y dependiente
# ---------------------------------------------------------------------------------------------
# predictoras -> presión, U_Wind, V_Wind y dependiente -> Temp_Max
X = CUCAO[['Pres', 'U_Wind', 'V_Wind']]  
y = CUCAO['Temp_Max']  

# estandarizar datos 
# ---------------------------------------------------------------------------------------------
sc = StandardScaler()
X_std = sc.fit_transform(X)

# modelo de RLM 
# ---------------------------------------------------------------------------------------------
# agregar una constante para el término independiente
X_std = sm.add_constant(X_std)
# se dividen los datos en grupo de entrenamiento y prueba (90%-10%)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, train_size=0.9, shuffle=False)

# se ajusta el modelo de regresión lineal múltiple
modelo = sm.OLS(y_train, X_train).fit()

# verifico
print(modelo.summary())

# predicciones con el conjunto de prueba
# ---------------------------------------------------------------------------------------------

y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)

# ---------------------------------------------------------------------------------------------

# grafico de los resultados observados vs. predichos
# plt.figure(figsize=(10, 6))
# plt.scatter(y_train, y_pred_train, label='Entrenamiento', color='blue')
# plt.scatter(y_test, y_pred_test, label='Prueba', color='red')
# plt.plot([min(y), max(y)], [min(y), max(y)], color='black', linestyle='--', label='Perfecto')
# plt.xlabel('Valores Observados [°C]')
# plt.ylabel('Valores Predichos [°C]')
# plt.title('Valores Observados vs. Predichos - Cucao (Temp_Max)')
# plt.legend()
# plt.grid(True)
# plt.show()

# histograma para los residuos
residuos = y_pred_train - y_train
plt.figure(figsize=(10, 6))
plt.hist(residuos, bins=15, edgecolor='black')
plt.title('Distribución de los Residuos')
plt.xlabel('Residuos [°C]')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# evaluación del rendimiento por medio de estadisticos 
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = modelo.rsquared
r2_test = r2_score(y_test, y_pred_test)
#verificamos en terminal 
print(f"Mean Squared Error (Entrenamiento): {mse_train:.4f}")
print(f"Mean Squared Error (Prueba): {mse_test:.4f}")
print(f"R² (Entrenamiento): {r2_train:.4f}")
print(f"R² (Prueba): {r2_test:.4f}")

# analizar la Multicolinealidad con VIF
# constante para VIF
X_vif = sm.add_constant(CUCAO[['Temp_Max', 'Pres', 'U_Wind', 'V_Wind']])

#calcular VIF para cada variable
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

# verificar el VIF
print("\nAnálisis de VIF (factor de fnflacioon de la varianza):")
print(vif_data)
