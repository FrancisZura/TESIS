# ARCH1 LUEGO DE DESCARGA DE DATOS GFS
# 
# este archivo es para trabajar los archivos grib luego de la descarga de estos, 
# obteniendo las variables temp_2m, presion, u_wind y v_wind
#----------------------------------------------------------------------------------------------------------------
import pygrib
import numpy as np
import os
#----------------------------------------------------------------------------------------------------------------
nombre_mes = "202004"  
num_mes_anio = "202004"  
# directorio archivos GRIB
directory = r'C:\Users\fzura\Desktop\2024-2\HP\descargas1'
# coordenadas 
melinka_coords = (-43.897, -73.75)
cucao_coords = (-42.645, -74.073)
#----------------------------------------------------------------------------------------------------------------
# función para el punto mas cercano en el grid
def find_nearest_index(lats, lons, lat_pt, lon_pt):
    dist_sq = (lats - lat_pt)**2 + (lons - lon_pt)**2
    minindex = dist_sq.argmin()
    return np.unravel_index(minindex, lats.shape)
# temperaturas maximas, minimas y medias
melinka_daily_data = {'temp_max': [], 'temp_min': [], 'temp_mean': [], 'pres': [], 'u_wind': [], 'v_wind': []}
cucao_daily_data = {'temp_max': [], 'temp_min': [], 'temp_mean': [], 'pres': [], 'u_wind': [], 'v_wind': []}
# ir por los archivos del directorio que correspondan al mes y año 
files_mes = [f for f in os.listdir(directory) if f.startswith('gfs.0p25.' + num_mes_anio)]

# verificar si se encontraron archivos
if not files_mes:
    print("No se encontraron archivos para el mes " + nombre_mes + " en el directorio.")
else:
    print(f"Procesando {len(files_mes)} archivos de {nombre_mes} {num_mes_anio}...")

# agrupar archivos por día
dias_mes = range(1, 32)  # Cambia el rango según el mes si es necesario (por ejemplo, para febrero)

for day in dias_mes:
    daily_melinka_temp = []
    daily_cucao_temp = []
    daily_melinka_data = {'pres': [], 'u_wind': [], 'v_wind': []}
    daily_cucao_data = {'pres': [], 'u_wind': [], 'v_wind': []}

    for hour in ['00', '06', '12', '18']:
        file = f'gfs.0p25.{num_mes_anio}{day:02d}{hour}.f000.grib2'
        filepath = os.path.join(directory, file)

        if not os.path.exists(filepath):
            print(f"Archivo {file} no encontrado.")
            continue

        try:
            print(f"Procesando archivo: {filepath}")
            grbs = pygrib.open(filepath)
            lats, lons = grbs[1].latlons() 

            # calcular los indices mas cercanos una sola vez por archivo
            melinka_idx = find_nearest_index(lats, lons, *melinka_coords)
            cucao_idx = find_nearest_index(lats, lons, *cucao_coords)

            try:
                pressure = grbs.select(name='Surface pressure')[0].values / 100.0  # en hPa
            except IndexError:
                pressure = None
                print("Presion no disponible en el archivo.")

            try:
                temp_2m = grbs.select(name='2 metre temperature')[0].values - 273.15  # en °C
            except IndexError:
                temp_2m = None
                print("Temperatura no disponible en el archivo.")
                
            try:
                u_wind = grbs.select(name='10 metre U wind component')[0].values  # en m/s
            except IndexError:
                u_wind = None
                print("Componente U del viento no disponible en el archivo.")

            try:
                v_wind = grbs.select(name='10 metre V wind component')[0].values  # en m/s
            except IndexError:
                v_wind = None
                print("Componente V del viento no disponible en el archivo.")

            # agregar datos a melinka y cucaco
            if temp_2m is not None:
                daily_melinka_temp.append(temp_2m[melinka_idx])
                daily_cucao_temp.append(temp_2m[cucao_idx])

            if pressure is not None:
                daily_melinka_data['pres'].append(pressure[melinka_idx])
                daily_cucao_data['pres'].append(pressure[cucao_idx])

            if u_wind is not None:
                daily_melinka_data['u_wind'].append(u_wind[melinka_idx])
                daily_cucao_data['u_wind'].append(u_wind[cucao_idx])

            if v_wind is not None:
                daily_melinka_data['v_wind'].append(v_wind[melinka_idx])
                daily_cucao_data['v_wind'].append(v_wind[cucao_idx])

            grbs.close()

        except Exception as e:
            print(f"Error procesando archivo {filepath}: {e}")
            continue

    # calcular temperaturas máxima, minima y media diarias 
    if daily_melinka_temp:
        melinka_daily_data['temp_max'].append(np.max(daily_melinka_temp))
        melinka_daily_data['temp_min'].append(np.min(daily_melinka_temp))
        melinka_daily_data['temp_mean'].append(np.mean(daily_melinka_temp))
        melinka_daily_data['pres'].append(np.mean(daily_melinka_data['pres']))
        melinka_daily_data['u_wind'].append(np.mean(daily_melinka_data['u_wind']))
        melinka_daily_data['v_wind'].append(np.mean(daily_melinka_data['v_wind']))

    if daily_cucao_temp:
        cucao_daily_data['temp_max'].append(np.max(daily_cucao_temp))
        cucao_daily_data['temp_min'].append(np.min(daily_cucao_temp))
        cucao_daily_data['temp_mean'].append(np.mean(daily_cucao_temp))
        cucao_daily_data['pres'].append(np.mean(daily_cucao_data['pres']))
        cucao_daily_data['u_wind'].append(np.mean(daily_cucao_data['u_wind']))
        cucao_daily_data['v_wind'].append(np.mean(daily_cucao_data['v_wind']))

# Guardar los datos diarios en archivos de texto
output_dir = r'C:\Users\fzura\Desktop\2024-2\HP\DATOS_6var'
try:
    melinka_output_file = os.path.join(output_dir, f'melinka_temp_min_max_avg_{nombre_mes}.txt')
    cucao_output_file = os.path.join(output_dir, f'cucao_temp_min_max_avg_{nombre_mes}.txt')

    with open(melinka_output_file, 'w') as f:
        f.write(f"Datos diarios para Melinka (Latitud: {melinka_coords[0]}, Longitud: {melinka_coords[1]})\n")
        f.write("Día    Temp_Mean (°C)    Temp_Max (°C)    Temp_Min (°C)    Pres (hPa)    Viento U (m/s)    Viento V (m/s)\n")
        for day in range(len(melinka_daily_data['temp_mean'])):
            f.write(f"{day+1:02d}    {melinka_daily_data['temp_mean'][day]:.2f}    {melinka_daily_data['temp_max'][day]:.2f}    "
                    f"{melinka_daily_data['temp_min'][day]:.2f}    {melinka_daily_data['pres'][day]:.2f}    "
                    f"{melinka_daily_data['u_wind'][day]:.2f}    {melinka_daily_data['v_wind'][day]:.2f}\n")

    with open(cucao_output_file, 'w') as f:
        f.write(f"Datos diarios para Cucao (Latitud: {cucao_coords[0]}, Longitud: {cucao_coords[1]})\n")
        f.write("Día    Temp_Mean (°C)    Temp_Max (°C)    Temp_Min (°C)    Pres (hPa)    Viento U (m/s)    Viento V (m/s)\n")
        for day in range(len(cucao_daily_data['temp_mean'])):
            f.write(f"{day+1:02d}    {cucao_daily_data['temp_mean'][day]:.2f}    {cucao_daily_data['temp_max'][day]:.2f}    "
                    f"{cucao_daily_data['temp_min'][day]:.2f}    {cucao_daily_data['pres'][day]:.2f}    "
                    f"{cucao_daily_data['u_wind'][day]:.2f}    {cucao_daily_data['v_wind'][day]:.2f}\n")

    print(f"Datos guardados correctamente en {output_dir} para el mes de {nombre_mes}.")
except Exception as e:
    print(f"Error al guardar los datos: {e}")
