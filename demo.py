#!/usr/bin/env python3

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import collections
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# visualize training data
# http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for index, item in enumerate(X):
    xs = item[0]
    ys = item[1]
    zs = item[2]

    color = 'r' if Y[index] == 'male' else 'b'
    m = '^' if Y[index] == 'male' else 'o'
    ax.scatter(xs, ys, zs, c=color, marker=m)

ax.set_xlabel('height')
ax.set_ylabel('weight')
ax.set_zlabel('shoe size')

# plt.show() interrupts program execution, user must close window to continue
# plt.show()
plt.savefig('people.png')

# train and predict

NamedClassifier = collections.namedtuple('NamedClassifier', 'name classifier')

# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
named_classifiers = [NamedClassifier("Nearest Neighbors", KNeighborsClassifier(3)),
                     NamedClassifier("Linear SVM", SVC(kernel="linear", C=0.025)),
                     NamedClassifier("Decision Tree", DecisionTreeClassifier()),
                     NamedClassifier("Naive Bayes", GaussianNB())
                     ]

for named_classifier in named_classifiers:
    # train classifier
    classifier = named_classifier.classifier.fit(X, Y)

    prediction = classifier.predict([[190, 70, 43]])
    print("{} predicts {}".format(named_classifier.name, prediction))
