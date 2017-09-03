from sklearn import tree

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3

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

plt.show()

# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)

prediction = clf.predict([[190, 70, 43]])

# CHALLENGE compare their reusults and print the best one!

print(prediction)
