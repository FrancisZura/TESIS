import pandas as pd
import numpy as np

file_path = r'C:\Users\fzura\Desktop\2024-2\HP\DATOS_IFOP\Datos_Emeteo_Francisca\Emeteo_Cucao_Francisca.csv'
df = pd.read_csv(file_path)

# arreflamos nombres de las columnas datos observaciones porque no estaban el de fechas
df.columns = ['Datetime', 'Direccion', 'Magnitud', 'Racha', 'T_2m', 'p_sfc']
# pasar columna de fechas a datetime
df['Datetime'] = pd.to_datetime(df['Datetime'])
# convertir las columnas de Dirección y Magnitud de viento en componentes U y V
df['U_Wind'] = -df['Magnitud'] * np.sin(np.radians(df['Direccion']))
df['V_Wind'] = -df['Magnitud'] * np.cos(np.radians(df['Direccion']))
# AQUI SE FILTRA EL MES DE ELECCION
df_enero = df[(df['Datetime'] >= '2021-01-01') & (df['Datetime'] < '2021-02-01')]
# agrupar los datos por día y calcular :temperatura maxima y minima diaria,promedio 
# diario de temperatura, presión y componentes del viento
df_diario = df_enero.resample('D', on='Datetime').agg({
    'T_2m': ['max', 'min', 'mean'],
    'p_sfc': 'mean',
    'U_Wind': 'mean',
    'V_Wind': 'mean'
})
# renombrar columnas
df_diario.columns = ['Temp_Max', 'Temp_Min', 'Temp_Mean', 'Pres_Mean', 'U_Wind_Mean', 'V_Wind_Mean']
# guardar archivo de texto
output_file_path = r'C:\Users\fzura\Desktop\2024-2\HP\Mains2\Mains2-1\Emeteo_Cucao_Enero_2021_Procesado.txt'
df_diario.to_csv(output_file_path, sep='\t', index=True)
#verifico
print(f"Datos diarios de enero 2021 guardados correctamente en {output_file_path}")
