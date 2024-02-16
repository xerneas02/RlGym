import os
import random
import pandas as pd
import re



data_directory = 'DataState'
random_csv_file = random.choice(os.listdir(data_directory))
random_csv_path = os.path.join(data_directory, random_csv_file)

random_data_file = open(random_csv_path, 'r')
random_data      = random_data_file.read()
random_data_file.close()

random_data = random_data.split("\n")
for i in range(len(random_data)):
    random_data[i] = random_data[i].split(",")


random_data[0] = [re.sub(r'^(?<=\.)\.|\.(\d+)$', '', s) for s in random_data[0]]

column = [(random_data[0][i], random_data[1][i]) for i in range(len(random_data[1]))]


player1 = None
player2 = None
for name, _ in column:
    if name != "ball" and name != "game":
        if player1 == None:
            player1 = name
        elif name != player1:
            player2 = name 


row = random.randint(2, len(random_data))

info_player1 = {}
info_player2 = {}
info_ball    = {}

for i in range(len(random_data[row])):
    if   column[i][0] == player1:
        info_player1[column[i][1]] = random_data[row][i]
    elif column[i][0] == player2:
        info_player2[column[i][1]] = random_data[row][i]
    elif column[i][0] == 'ball' :
        info_ball[column[i][1]]    = random_data[row][i] 

