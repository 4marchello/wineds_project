# ГОЛОВНИЙ СКРИПТ: main.py
import os
import sys

shlyah=os.path.join(os.path.expanduser("~"),"Desktop","ds_final_project")
if shlyah not in sys.path:
    sys.path.append(shlyah)

import pandas as pd
from ds_project.utils import zavantazhyty
from ds_project.data.preprocessing import ochystyty, rozdylyty
from ds_project.features.build_features import stvorytyoznaky, masshtab, vidbir
from ds_project.models.train_model import navchaty, ocinyty

def golovna():
    print("--- 1. Первинне дослідження даних ---")
    dani=zavantazhyty()
    print(f"Forma datasetu: {dani.shape}")

    print("\n--- 2. Preprocessing (Очищення та спліт) ---")
    dani_clean=ochystyty(dani)
    xtrain,xtest,ytrain,ytest=rozdylyty(dani_clean)
    print(f"Rozmirnist train: {xtrain.shape}, test: {xtest.shape}")

    print("\n--- 3. Feature Engineering & Selection ---")
    xtrain_fe=stvorytyoznaky(xtrain)
    xtest_fe=stvorytyoznaky(xtest)
    print(f"Kilkist oznak pislya stvorennya novyh: {xtrain_fe.shape[1]}")
    
    xtrain_sk,xtest_sk=masshtab(xtrain_fe,xtest_fe)
    xtrain_final,xtest_final=vidbir(xtrain_sk,xtest_sk)
    print(f"Kilkist oznak pislya vidboru (Selection): {xtrain_final.shape[1]}")

    print("\n--- 4. Навчання моделі та Оцінка якості ---")
    model=navchaty(xtrain_final,ytrain)
    rezultaty=ocinyty(model,xtest_final,ytest)

    for m,v in rezultaty.items():
        if m!="matrix":
            print(f"{m.upper()}: {v:.4f}")
        else:
            print("\nConfusion Matrix:")
            print(pd.DataFrame(v,columns=["Pred 0","Pred 1","Pred 2"],index=["Actual 0","Actual 1","Actual 2"]))

if __name__=="__main__":
    golovna()