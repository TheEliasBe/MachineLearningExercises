# implementation of a k-NN Classifier for the iris flower data set

import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


iris = pd.read_csv('datasets\iris.csv')
training_frac = 0.7 # fraction of data used for training

iris_train = iris.sample(frac=training_frac,random_state=0)
iris_train_x = iris_train[['petal_length','sepal_length']]
iris_train_y = iris_train['species']
iris_test = iris.drop(iris_train.index)
iris_test_x = iris_test[['petal_length','sepal_length']]
iris_test_y = iris_test['species']

# distance function by 2d euclidean distance
def d(x,y):
    return ((x[0]-y[0])**2+(x[1]-y[1])**2)**0.5

# my own nearest neighbor implementation
def nn(x,k):
    # generate priority queue of k nearest neighbours
    distances = pd.DataFrame(columns=["distance","species"])
    for index,e in iris_train.iterrows():
        dis = d(x,(e['petal_length'],e['sepal_length']))
        distances = distances.append(pd.DataFrame([{"distance": dis,"species":e['species']}]))
    distances = distances.sort_values(by=['distance'])
    distances = distances.head(k)

    # count types of nearest neighbors
    n = [0,0,0]
    for i,y in distances.iterrows():
        if y["species"] == "setosa":
            n[0] += 1
        elif y["species"] == "virginica":
            n[1] += 1
        elif y["species"] == "versicolor":
            n[2] += 1

    # classify based on counted neighbours.
    if n[0] > (n[1] & n[2]):
        return "setosa"
    elif n[1] > (n[0] & n[2]):
        return "virginica"
    elif n[2] > (n[1] & n[0]):
        return "versicolor"
    elif n[0] == n[1]:
        if random.random() > 0.5:
            return "setosa"
        else:
            return "virginica"
    elif n[1] == n[2]:
        if random.random() > 0.5:
            return "versicolor"
        else:
            return "virginica"
    elif n[0] == n[2]:
        if random.random() > 0.5:
            return "setosa"
        else:
            return "versicolor"


def main(k):
    error = 0
    for index,e in iris_test.iterrows():
        x1 = e['petal_length']
        x2 = e['sepal_length']
        y_bar = nn((x1,x2),k) # prediction
        y = e['species'] # actual
        if y_bar != y:
            error += 1
    return error/iris_test.shape[0]


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(iris_train_x, iris_train_y)
y_pred = knn.predict(iris_test_x)
print("SKLearn Implementation Accuracy: " + accuracy_score(iris_train_y,y_pred))
print("Custom Implementation Accuracy: " + 1-main(5))


