# МОДУЛЬ: build_features. Ознаки, масштабування, селекція.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def stvorytyoznaky(df):
    df=df.copy()
    matr=df.to_numpy()
    df["oznaka_sum"]=matr[:,0]+matr[:,1]
    df["oznaka_vidnosna"]=matr[:,2]/(matr[:,3]+1e-5)
    df["oznaka_log"]=np.log1p(np.abs(matr[:,4]))
    return df

def masshtab(xtrain,xtest):
    skaler=StandardScaler()
    numcols=xtrain.select_dtypes(include=["float64","int64"]).columns
    xtrain_sk=xtrain.copy()
    xtest_sk=xtest.copy()
    xtrain_sk[numcols]=skaler.fit_transform(xtrain[numcols])
    xtest_sk[numcols]=skaler.transform(xtest[numcols])
    return xtrain_sk,xtest_sk

def vidbir(xtrain,xtest):
    kor=xtrain.corr().abs()
    vydat=[]
    kolonky=kor.columns
    
    for i in range(len(kolonky)):
        for j in range(i):
            if kor.iloc[i,j]>0.85:
                kol_name=kolonky[i]
                if kol_name not in vydat:
                    vydat.append(kol_name)
                    
    xtrain_vyd=xtrain.drop(columns=vydat)
    xtest_vyd=xtest.drop(columns=vydat)
    return xtrain_vyd,xtest_vyd
