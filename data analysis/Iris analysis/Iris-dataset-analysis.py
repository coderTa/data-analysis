from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

with open('Iris.csv', 'r') as iris_data:
    lines = iris_data.readlines()
    del lines[0]
    split_lines = []

    for i in range(len(lines)):
        split_lines.append(lines[i].split(','))

#print(split_lines)

x = []
y = []

for i in range(len(split_lines)):

    tx = []

    for j in range(1, 5):
        tx.append(float(split_lines[i][j]))

    if split_lines[i][5] == 'Iris-setosa\n':
        y.append([1, 0, 0])
    elif split_lines[i][5] == 'Iris-versicolor\n':
        y.append([0, 1, 0])
    else:
        y.append([0, 0, 1])

    x.append(tx)

#print(x, y)

x = np.array(x)
y = np.array(y)

classifier = KNeighborsClassifier()
classifier.fit(x, y)

results = classifier.score(x, y)

#print(results)

matrix = confusion_matrix(np.argmax(y, axis = -1), np.argmax(classifier.predict(x), axis = -1))

#print(matrix)

plt.imshow(matrix)
plt.show()