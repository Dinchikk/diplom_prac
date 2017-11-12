import numpy as p
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
plt.rc('font', family='Verdana')


from sklearn.datasets import load_iris
iris_dataset = load_iris()
print("Ключи: \n{}".format(iris_dataset.keys()))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test = train_test_split(iris_dataset['data'] ,
                        iris_dataset['target'],random_state = 0)

print("размер массива X_train: {}".format(X_train.shape))
print("размер массива X_test: {}".format(X_test.shape))
print("размер массива y_train: {}".format(y_train.shape))
print("размер массива y_test: {}".format(y_test.shape))


#iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
#grr = pd.scatter_matrix(iris_dataframe , c=y_train , figsize=(15,15),marker='o',
#                       hist_kwds={'bins':20},s=60 ,alpha=.8, cmap=mglearn.cm3)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array([[5 , 2.9 , 1 , 0.2]])
prediction = knn.predict(X_new)
print("Прогноз: {}".format(
        iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("Точность : {:.2f}".format(np.mean(y_pred == y_test)))
print("Точность 2 :{:.2f}".format(knn.score(X_test , y_test)))