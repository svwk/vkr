from multiprocessing import Pool
import numpy as np
import pandas as pd
from config import settings

# физических ядра
n_cores = settings.get_fresh('N_CORES')
pool = Pool(n_cores)

def apply_parallel(dataframe, func):
    # делим датафрейм на части
    df_split = np.array_split(dataframe, n_cores)
    # считаем метрики для каждого и соединяем обратно
    df = pd.concat(pool.map(func, df_split), ignore_index=True)
    return df

def apply_parallel_todict(dataframe, func_for_input, dunc_to_output):
    # делим датафрейм на части
    df_split = np.array_split(dataframe, n_cores)
    # считаем метрики для каждого и соединяем обратно
    result = dunc_to_output(pool.map(func_for_input, df_split))
    return result
