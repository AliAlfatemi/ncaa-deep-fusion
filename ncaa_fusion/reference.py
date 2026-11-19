"""Original five-model cognitive diversity calculations from the NCAA script."""
import pandas as pd

def cognitiveD(raw):
    df2 = pd.DataFrame({"Model1" :[abs(raw.Model1.sum()-raw.Model1.sum()), abs(raw.Model2.sum()-raw.Model1.sum()), abs(raw.Model3.sum()-raw.Model1.sum()), abs(raw.Model4.sum()-raw.Model1.sum()), abs(raw.Model5.sum()-raw.Model1.sum())],
                       "Model2" :[abs(raw.Model1.sum()-raw.Model2.sum()), abs(raw.Model2.sum()-raw.Model2.sum()) , abs(raw.Model3.sum()-raw.Model2.sum()), abs(raw.Model4.sum()-raw.Model2.sum()), abs(raw.Model5.sum()-raw.Model2.sum())],
                       "Model3" :[abs(raw.Model1.sum()-raw.Model3.sum()), abs(raw.Model2.sum()-raw.Model3.sum()), abs(raw.Model3.sum()-raw.Model3.sum()), abs(raw.Model4.sum()-raw.Model3.sum()), abs(raw.Model5.sum()-raw.Model3.sum())],
                       "Model4": [abs(raw.Model1.sum()-raw.Model4.sum()), abs(raw.Model2.sum()-raw.Model4.sum()), abs(raw.Model3.sum()-raw.Model4.sum()), abs(raw.Model4.sum()-raw.Model4.sum()), abs(raw.Model5.sum()-raw.Model4.sum())],
                       "Model5": [abs(raw.Model1.sum()-raw.Model5.sum()), abs(raw.Model2.sum()-raw.Model5.sum()), abs(raw.Model3.sum()-raw.Model5.sum()), abs(raw.Model4.sum()-raw.Model5.sum()), abs(raw.Model5.sum()-raw.Model5.sum())]},
                       index = ["Model1", "Model2", "Model3", "Model4", "Model5"])
    df2 = df2[["Model1", "Model2", "Model3", "Model4", "Model5"]]
    return df2

def diversityS(raw):
    strength_A = (abs(raw.Model1.sum()-raw.Model2.sum()) + abs(raw.Model1.sum()-raw.Model3.sum()) + abs(raw.Model1.sum()-raw.Model4.sum()) + abs(raw.Model1.sum()-raw.Model5.sum()))/4
    strength_B = (abs(raw.Model2.sum()-raw.Model1.sum()) + abs(raw.Model2.sum()-raw.Model3.sum()) + abs(raw.Model2.sum()-raw.Model4.sum()) + abs(raw.Model2.sum()-raw.Model5.sum()))/4
    strength_C = (abs(raw.Model3.sum()-raw.Model1.sum()) + abs(raw.Model3.sum()-raw.Model2.sum()) + abs(raw.Model3.sum()-raw.Model4.sum()) + abs(raw.Model3.sum()-raw.Model5.sum()))/4
    strength_D = (abs(raw.Model4.sum()-raw.Model1.sum()) + abs(raw.Model4.sum()-raw.Model2.sum()) + abs(raw.Model4.sum()-raw.Model3.sum()) + abs(raw.Model4.sum()-raw.Model5.sum()))/4
    strength_E = (abs(raw.Model5.sum()-raw.Model1.sum()) + abs(raw.Model5.sum()-raw.Model2.sum()) + abs(raw.Model5.sum()-raw.Model3.sum()) + abs(raw.Model5.sum()-raw.Model4.sum()))/4
    return strength_A, strength_B, strength_C, strength_D, strength_E
