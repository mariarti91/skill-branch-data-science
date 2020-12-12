import numpy as np
import pandas as pd

def calculate_data_shape(data):
    return data.shape

def take_columns(data):
    return data.columns

def calculate_target_ratio(data, target):
    mean = data[target].mean()
    return round(mean, 2)

def calculate_data_dtypes(data):
    return data.select_dtypes(include=[int, float, object]).count().sum()

def calculate_cheap_apartment(data):
    return data[data['price_doc'] <= 1000000].shape[0]

def calculate_squad_in_cheap_apartment(data):
    return int(data[data['price_doc'] <= 1000000]['full_sq'].mean())

def calculate_mean_price_in_new_housing(data):
    return int(data[(data['build_year'] >= 2010) & (data['num_room'] == 3)]['price_doc'].mean())

def calculate_mean_squared_by_num_rooms(data):
    return data.groupby('num_room')['full_sq'].mean().round(2)

def calculate_squared_stats_by_material(data):
    return data.groupby('material').agg({'material': ['min', 'max']}).round(2)

def calculate_crosstab(data):
    return data.groupby(['sub_area', 'product_type'])['price_doc'].mean().unstack().fillna(0).round(2)


if __name__ == '__main__':        
    # data = pd.DataFrame({'country': ['Kazakhstan', 'Russia', 'Belarus', 'Ukraine'],
    #                 'population': [17.04, 143.5, 9.5, 45.5],
    #                 'square': [2724902, 17125191, 207600, 603628]
    #                 }, index=['KZ', 'RU', 'BY', 'UA'])

    data = pd.read_csv('sberbank_housing_market.csv', sep = ',')                

    print(calculate_data_shape(data))
    print(take_columns(data))
    print(calculate_target_ratio(data, 'build_year'))
    print(calculate_data_dtypes(data))
    print(calculate_cheap_apartment(data))
    print(calculate_squad_in_cheap_apartment(data))
    print(calculate_mean_price_in_new_housing(data))
    print(calculate_mean_squared_by_num_rooms(data))
    print(calculate_squared_stats_by_material(data))
    print(calculate_crosstab(data))