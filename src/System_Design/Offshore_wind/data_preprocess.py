from keys import Files,Cols,DataDir,CACHE_DATA_DIR
import pandas as pd
import os

def load_raw_data():
    df_raw = pd.read_excel(
        os.path.join(
            DataDir.Raw,
            'Offshore_wind_power.xlsx'
        )
    )
    df_raw = df_raw.rename(
        columns = {
            '时间':Cols.time, '出力值(万kW)':Cols.wind
        }
    )
    df_raw[Cols.wind] = df_raw[Cols.wind]*10000/1000 # MW
    df_wind = df_raw
    return df_wind

def generate_simulation_df(wind_cap):
    df_simulation = load_raw_data()
    # 对风电的出力曲线进行缩放
    rated_raw_cap = 700.
    df_simulation[Cols.wind] = df_simulation[Cols.wind] / rated_raw_cap
    df_simulation[Cols.wind] = df_simulation[Cols.wind] * wind_cap # MW
    df_simulation[Cols.AWES] = 0 # MW
    # df_simulation[Cols.BoP] = 0 # MW，BoP功率，这个项目里主要指压缩机，现在被算在AWES里面
    df_simulation[Cols.water_purification] = 0 # MW，海水淡化功率
    df_simulation[Cols.hydrogen_production] = 0 # t，氢气生产量
    df_simulation[Cols.energy_storage] = 0 # MWh，储能容量
    df_simulation[Cols.energy_storage_power] = 0 # MW，储能功率
    df_simulation[Cols.curtailment_wind] = 0 # MW，弃风功率
    return df_simulation