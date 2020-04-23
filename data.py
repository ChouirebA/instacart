import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")


def read_orders():
    orders = pd.read_csv("C:/Users/anis_/Desktop/Data/Jupyter/Instacart/orders.csv")
    return orders[['order_id', 'user_id']]


def read_products():
    products = pd.read_csv("C:/Users/anis_/Desktop/Data/Jupyter/Instacart/products.csv")
    products = products.dropna(subset=['department_id'])
    # print(pd.unique(products['department_id']))
    return products[['product_id', 'department_id']]


def read_orders_products():
    orders_products = pd.read_csv("C:/Users/anis_/Desktop/Data/Jupyter/Instacart/order_products__prior.csv")
    # orders_train = pd.read_csv("C:/Users/anis_/Desktop/Data/Jupyter/Instacart/order_products__train.csv")
    # orders_products= pd.concat([orders_prior, orders_train], axis=0)
    return orders_products[['order_id', 'product_id']]


def user_table(nb_users):
    df_orders = read_orders()
    df_products = read_products()
    df_orders_products = read_orders_products()

    users = pd.merge(df_orders, df_orders_products, on='order_id')
    users = pd.merge(users, df_products, on='product_id')
    users = users[['order_id', 'user_id', 'department_id']]

    user_col = pd.unique(users['user_id']).tolist()
    user_col = user_col[:nb_users]
    rows = []

    for i in user_col:
        data = users[users['user_id'] == i]
        # data = data.sort_values(by = ['order_id'], ascending = True)
        order_col = pd.unique(data['order_id']).tolist()

        liste = []
        # liste_order = []
        for j in order_col:
            # liste_order.append(j)
            liste.append(data[data['order_id'] == j]['department_id'])

        rows.append([i, liste])
    df = pd.DataFrame(rows, columns=['user_id', 'department_id'])
    df = df['department_id']

    return df 
    