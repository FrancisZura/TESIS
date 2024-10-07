import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#----------------------------------------------------------------------------------------------------
# datos obs - gfs -> unir con script : uni_obs_gfs.py 
file_path = r'C:\Users\fzura\Desktop\2024-2\HP\RLM_1\datos_obs_gfs.csv'
df = pd.read_csv(file_path)
# filtrar las columnas de interes -> predictante:Temp_Mean_obs y predictores de GFS
df = df[['Temp_Mean_obs', 'Temp_Mean_gfs', 'Pres_gfs', 'Viento_U_gfs', 'Viento_V_gfs']]
#----------------------------------------------------------------------------------------------------
# matriz de correlación entre las variables predictoras
correlation_matrix = df[['Temp_Mean_gfs', 'Pres_gfs', 'Viento_U_gfs', 'Viento_V_gfs']].corr()
# grafica 
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', vmin=-1, vmax=1)
plt.title('Matriz de Correlación de las Variables Predictor (GFS)')
plt.savefig(r'C:\Users\fzura\Desktop\2024-2\HP\RLM_1\matriz_corr.png')  # Guardar la figura
plt.show()
#----------------------------------------------------------------------------------------------------
# definir las variables predictoras (X) y la variable predictante (y)
X = df[['Temp_Mean_gfs', 'Pres_gfs', 'Viento_U_gfs', 'Viento_V_gfs']]
y = df['Temp_Mean_obs']
# dividir en entrenamiento (90%) y prueba (10%)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=False)
# añadir una columna de unos a las variables predictoras para el intercepto del modelo, diagonal
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
# modelo RLM
modelo = sm.OLS(y_train, X_train)
resultado = modelo.fit()
# guardar el resumen del modelo en un archivo de txt
with open(r'C:\Users\fzura\Desktop\2024-2\HP\RLM_1\regresion.txt', 'w') as f:
    f.write(resultado.summary().as_text())
# predicciones con los datos de entrenamiento
pred_train = resultado.predict(X_train)
# predicciones con los datos de prueba
pred_test = resultado.predict(X_test)
# graficar 
plt.figure(figsize=(10, 6))
plt.plot(df['Temp_Mean_obs'].index[:len(y_train)], y_train, label='Observado (Entrenamiento)', color='blue')
plt.plot(df['Temp_Mean_obs'].index[:len(y_train)], pred_train, label='Predicción (Entrenamiento)', color='red')
plt.plot(df['Temp_Mean_obs'].index[len(y_train):], y_test, label='Observado (Prueba)', color='green')
plt.plot(df['Temp_Mean_obs'].index[len(y_train):], pred_test, label='Predicción (Prueba)', color='orange')
plt.xlabel('Fecha')
plt.ylabel('Temperatura Media [°C]')
plt.title('Regresión Lineal Múltiple - GFS vs Observaciones')
plt.legend()
plt.grid(True)
plt.savefig(r'C:\Users\fzura\Desktop\2024-2\HP\RLM_1\rlm_results.png')  # Guardar la figura
plt.show()
# calcular metricas
mse_train = mean_squared_error(y_train, pred_train)
mse_test = mean_squared_error(y_test, pred_test)
r2_train = r2_score(y_train, pred_train)
r2_test = r2_score(y_test, pred_test)
# guardar las métricas en un archivo de texto
with open(r'C:\Users\fzura\Desktop\2024-2\HP\RLM_1\metricas.txt', 'w') as f:
    f.write(f'MSE Train: {mse_train}\n')
    f.write(f'MSE Test: {mse_test}\n')
    f.write(f'R² Train: {r2_train}\n')
    f.write(f'R² Test: {r2_test}\n')
# verificar
print(f'MSE Train: {mse_train}')
print(f'MSE Test: {mse_test}')
print(f'R² Train: {r2_train}')
print(f'R² Test: {r2_test}')
#----------------------------------------------------------------------------------------------------