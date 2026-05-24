import pandas as pd
import numpy as np
from ds_project.data.preprocessing import ochystyty
from ds_project.features.build_features import stvorytyoznaky

def test_ochystyty_handles_nan():
    # Тест перевіряє, чи заповнюються пропуски (NaN)
    df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [2.0, 4.0, 6.0]})
    df_clean = ochystyty(df)
    assert df_clean["A"].isnull().sum() == 0, "Пропуски не були заповнені!"

def test_stvorytyoznaky_adds_columns():
    # Тест перевіряє, чи додаються нові 3 ознаки
    df = pd.DataFrame({
        "col1": [1, 2], "col2": [3, 4], "col3": [5, 6], 
        "col4": [7, 8], "col5": [9, 10]
    })
    df_features = stvorytyoznaky(df)
    assert "oznaka_sum" in df_features.columns
    assert "oznaka_vidnosna" in df_features.columns
    assert "oznaka_log" in df_features.columns