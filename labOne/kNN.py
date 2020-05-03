import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def thirdTask(dataPath: str):
    # Чтение данных из файла в dataset
    dataset = pd.read_csv(dataPath, sep=",")

    # Удаление столбца Id
    dataset.drop(['Id'], axis='columns', inplace=True)

    # Разделение на признаки и метки
    feature = dataset.iloc[:, :-1]
    label = dataset['Type']
    print(feature)
    print(label)

    # Деление данных на обучающие и тестовые
    trainSize = 0.7
    testSize = 1 - trainSize
    feature_train, feature_test, label_train, label_test = train_test_split(feature, label, train_size=trainSize, test_size=testSize)

    # Создание и обучение экземпляра классификатора
    knnCl = KNeighborsClassifier(n_neighbors=5)
    knnCl.fit(feature_train, label_train)

    # Тестирование классификатора
    knnResult = knnCl.predict(feature_test)

    # Оценка точности классификатора
    knnAccuracy = accuracy_score(knnResult, label_test)
    print(knnAccuracy)


thirdTask("data\glass.csv")
