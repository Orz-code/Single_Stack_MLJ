import requests
import urllib.request
import chardet
from tqdm import tqdm
from keys import *
import os
import pandas as pd
import json
import datetime
import re

def request_2345_web(selected_month = '202306',city_code = 60897):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) \
        Chrome/77.0.3865.120 Safari/537.36"
    }
    todo_url = f"http://tianqi.2345.com/t/wea_history/js/{selected_month}/{city_code}_{selected_month}.js"
    try:
        response = requests.get(todo_url,headers=headers)
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    # 确保请求成功
    if response.status_code == 200:
        # 检测内容编码
        result = chardet.detect(response.content)
        encoding = result['encoding']
        # 使用检测到的编码解码内容
        text_data = response.content.decode(encoding)
    else:
        raise SystemExit('request未能读成功')
    return text_data


def process_2345_web_text(text):
    text = text.replace("var weather_str=", "").rstrip(";")

    # 替换单引号为双引号
    text = text.replace("'", '"')
    text = text.strip()

    # 使用 replace 方法将单引号替换为双引号
    text = text.replace("'", '"')

    # 修正未引用的键名，使用正则表达式将未引用的键名用双引号包围
    text = re.sub(r'(\w+):', r'"\1":', text)

    # 去掉末尾的空对象
    text = text.replace(',{}', '')
    contents = json.loads(text)
    return contents

def process_2345_web_contents(contents):
    res_list = []
    for cur_line in contents['tqInfo']:
        date = cur_line['ymd']
        temp_max = cur_line['bWendu']
        temp_min = cur_line['yWendu']
        weather = cur_line['tianqi']
        try:
            wind_level = int(cur_line['fengli'].split('级')[0])
        except ValueError:
            wind_level = 0 # 如果出错应该是微风，默认是0
        try:
            aqi = cur_line['aqi']
        except KeyError:
            aqi = 0 # 如果出错可能是当前月份没有aqi，就填充0
        res_list.append(
            [
                date,
                temp_max,
                temp_min,
                weather,
                wind_level,
                aqi,
            ]
        )
    df_cur = pd.DataFrame(
        data = res_list,
        columns= [
            Cols.date_time,
            'temp_max',
            'temp_min',
            'weather',
            'wind_level',
            'aqi'
        ]
    )
    return df_cur


def get_history_temps_month(selected_month,city_code = 60897):
    """
        这个函数可以按照月份需求读取2345天气网上的历史数据，并搜索得到历史上某一天的最高温和最低温
        months输入为列表，格式为%yyyy%mm
        原始方法：https://yonniye.com/archives/11.html
    """
    text = request_2345_web(
        selected_month=selected_month,
        city_code=city_code,
    )
    contents = process_2345_web_text(text)
    df_cur = process_2345_web_contents(contents)
    return df_cur

def generate_month_range(start_month, end_month):
    
    # 将起始月份和结束月份转换为 datetime 对象
    start_date = pd.to_datetime(start_month)
    end_date = pd.to_datetime(end_month)
    
    # 初始化一个列表来存储结果
    month_list = []

    # 使用 while 循环生成月份范围
    current_date = start_date
    while current_date <= end_date:
        # 将当前月份格式化为 YYYYMM 格式并添加到列表中
        month_list.append(current_date.strftime("%Y%m"))
        # 增加一个月
        next_month = current_date.month + 1 if current_date.month < 12 else 1
        next_year = current_date.year if current_date.month < 12 else current_date.year + 1
        current_date = datetime.datetime(next_year, next_month, 1)
    
    return month_list

def get_history_temps(
        start_month = '2023-06',
        end_month = '2024-06',
        city_code = 60897
    ):
    month_list = generate_month_range(start_month, end_month)
    res_list = []
    for selected_month in tqdm(month_list):
        try:
            df_cur = get_history_temps_month(selected_month,city_code)
            res_list.append(df_cur)
        except:
            print(selected_month)
    df_res = pd.concat(
        res_list
    )
    return df_res

def main():
    df_res = get_history_temps()
    df_res.to_csv(
        os.path.join(
            DataDir.Raw,
            '哈得历史天气_2023_06_2024_06.csv'
        ),
        encoding='utf-8-sig'
    )
    print('History local temperature saved at: '+os.path.join(
            DataDir.Raw,
            '哈得历史天气_2023_06_2024_06.csv'
        ))

if __name__=='__main__':
    main()
