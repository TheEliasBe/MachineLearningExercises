import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('./datasets/ClassDecision.csv')

plt.scatter(df.x,df.y,c=df.label)
# plt.show()

class Node:

    def __init__(self,d):
        self.data = d
        pass

    def get_distribution(self):
        prob1 = self.data.label.value_counts()[1]
        prob2 = self.data.label.value_counts()[2]
        prob3 = self.data.label.value_counts()[3]
        return prob1,prob2,prob3

    def gini(self):
        gini = 0
        classes = self.data.label.unique()
        for c in classes:
            prob = self.data.label.value_counts()[c] / len(self.data.label)
            gini += prob**2
        return 1-gini

    def gini_x1(self,x1):
        gini = 0
        classes = self.data.label.unique()
        for c in classes:
            prob = self.data.label.value_counts()[c] / len(self.data.label)
            gini += prob ** 2
        return 1 - gini

    # compute entropy at a node
    def entropy(self):
        entropy = 0
        classes = self.data.label.unique()  # how many classes are there
        for c in classes:
            prob = self.data.label.value_counts()[c] / len(self.data.label)  # probability of classifying an object into a certain class
            entropy += prob * np.log2(prob)  # shannon entropy
        entropy *= -1
        return entropy


n = Node(df)
print(n.get_distribution())
print(n.entropy())
print(n.gini())