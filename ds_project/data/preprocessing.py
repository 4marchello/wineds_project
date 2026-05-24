# МОДУЛЬ: preprocessing. Очищення та спліт.
import pandas as pd
from sklearn.model_selection import train_test_split

def ochystyty(df):
    df=df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ["float64","int64"]:
                df[col]=df[col].fillna(df[col].median())
            else:
                df[col]=df[col].fillna(df[col].mode()[0])
    return df

def rozdylyty(df):
    # Беремо НАЙПЕРШУ колонку як цільову зміну (клас вина)
    persha_kolonka=df.columns[0]
    x=df.drop(columns=[persha_kolonka])
    y=df[persha_kolonka]
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
    return xtrain,xtest,ytrain,ytest
