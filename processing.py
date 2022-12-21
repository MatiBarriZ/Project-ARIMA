import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.time import TimeDelta
from astropy.timeseries import TimeSeries
from tqdm.auto import tqdm # Para la barra de progreso
import time 
import traces
import datetime
from datetime import timedelta


def read_df(path):
    data = pd.read_csv(path + '/metadata.csv') #Cambiar aquí la dirección por la que uds tengan 
    df = data[data['Class'] == 'RRab'].reset_index(drop=True)
    return df

    
def add_iso_time(df):
    times = []
    for i in range(len(df)):
        aux = Time(df['mjd'][i], format = 'mjd')
        times.append(pd.to_datetime(aux.to_value('iso')))
    df2 = pd.DataFrame()
    df2['mjd'] = df['mjd']
    df2['mag'] = df['mag'] 
    df2['err'] = df['err'] 
    df2['iso'] = times
    return(df2)

def get_objects(df, objects_path, num_objects_to_read):
    objects = dict()
    paths = df['Path']
    for i in tqdm(range(len(paths))):
        if i>= num_objects_to_read:
            break
        time.sleep(0.001)
        path = paths[i]
        data = pd.read_csv(objects_path+'/LCs/'+path)
        objects[path] = add_iso_time(data)
    return objects

def get_regular_linear_interpolation(data): #Esta funcion entrega el dataframe con la serie regularizada con interpolacion lineal
    ts  = traces.TimeSeries()
    for i in range(len(data)):
        ts[data['iso'][i]] = data['mag'][i]
    start = pd.to_datetime(data['iso'][0])
    end = pd.to_datetime(data['iso'][len(data)-1])
    regular_ts = ts.sample(sampling_period = (end-start)/len(data), start = start, end = end, interpolate = 'linear')
    regular_df = pd.DataFrame()
    regular_df['iso'] = [a[0] for a in regular_ts]
    regular_df['mag'] = [a[1] for a in regular_ts]
    return regular_df
def get_delta_t(data): #Esta funcion entrega una lista con los delta t
    gap = []
    for i in range(1,len(data)):
        val = (pd.to_datetime(data['iso'][i], utc = True) - pd.to_datetime(data['iso'][i-1], utc = True))/pd.Timedelta(hours = 1)
        gap.append(val)
    return gap
def plot_everything(data, path): #esta funcion plotea todo a partir de un dataframe (de la serie no regularizada)
    fig, ax = plt.subplots(2)
    fig.tight_layout(pad=4.0)
    ax[0].plot(data['iso'], data['mag'], 'k.', markersize=1, label = 'original series')
    ax[0].set_title(path)
    
    regular = get_regular_linear_interpolation(data)
    ax[0].plot(regular['iso'], regular['mag'], 'r*', markersize=1, label = 'regular series')
    ax[0].set_title(path)
    ax[0].set(xlabel="Time (iso)", ylabel="Magnitude")
    ax[0].legend()
    
    delta1 = get_delta_t(data)
    ax[1].plot(list(range(1, len(data))), get_delta_t(data), label = '$\Delta t$ original')
    ax[1].plot(list(range(1, len(regular))), get_delta_t(regular), label = '$\Delta t$ regularized')
    ax[1].set_title('Time gaps')
    ax[1].set(xlabel="Time (iso)", ylabel="Magnitude")
    ax[1].legend()
    plt.savefig('img' + path + '.svg')

