from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd


# Задание 1

def firstTask(trainSize: float, dataPath: str, dataType: str):
    # Чтение данных из файла в dataset
    testSize = 1 - trainSize
    if (dataType == "txt"):
        dataset = pd.read_csv(dataPath, sep=",", header=None)
    elif (dataType == "csv"):
        dataset = pd.read_csv(dataPath)
    print(dataset)

    # Разделение на признаки и метки
    feature = dataset.iloc[:, :-1]
    lable = dataset.iloc[:, 9]

    # Деление данных на обучающие и тестовые





