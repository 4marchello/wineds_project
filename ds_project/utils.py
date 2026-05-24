import os
import pandas as pd
import urllib.request
# В мене чомусь не працював урл тому щнайшов таке рішення
def zavantazhyty():
    baza=os.path.dirname(os.path.abspath(__file__))
    papka=os.path.join(baza,"data")
    os.makedirs(papka,exist_ok=True)
    shlyah=os.path.join(papka,"wine_types.csv")
    
    df=pd.read_csv(shlyah)
    return df
