import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm
import sklearn.tree
import sklearn.neural_network

with open("C:/Users/Ben Ta/Desktop/NBA_shot_stats/nba-shot-logs/shot_logs.csv", 'r') as d:
    lines = d.readlines()

dataset = []

for line in lines[1:]:
    split_line = line.replace(', ', '').split(',')

    if '' in split_line:
        continue

    location = split_line[2] == "H"

    shot_number = int(split_line[5])
    shot_number /= 25

    period = int(split_line[6])
    period /= 4

    #print(split_line)
    game_clock = split_line[7].split(':')
    #print(game_clock)
    game_clock = int(game_clock[0]) * 60 + int(game_clock[1])
    game_clock /= 720

    shot_clock = float(split_line[8])
    shot_clock /= 24

    shot_dist = float(split_line[11])
    shot_dist /= 40

    shot_result = split_line[13] == "made"

    datapoint = [location, shot_number, period, game_clock, shot_clock, shot_dist, shot_result]

    dataset.append(datapoint)

dataset = np.array(dataset)


shots_home = dataset[np.where(dataset[:, 0] == True)]
shots_away = dataset[np.where(dataset[:, 0] == False)]

print(len(np.where(shots_home[:, -1] == True)[0]) / len(shots_home))
print(len(np.where(shots_home[:, -1] == False)[0]) / len(shots_home))
print(len(np.where(shots_away[:, -1] == True)[0]) / len(shots_away))
print(len(np.where(shots_away[:, -1] == False)[0]) / len(shots_away))

#print(dataset[0])
#print(len(dataset))

print("Begin Training")
classifier = sklearn.neural_network.MLPClassifier()
classifier.fit(dataset[:10000, 0 : -1], dataset[:10000, -1])
print("End Training")

correct = classifier.predict(dataset[10000:15000, 0 : -1]) == dataset[10000:15000, -1]
print(correct)

print(np.sum(correct) / 5000)

#plt.hist2d(dataset[:, 5], dataset[:, 6], bins = [24, 2])
#plt.show()