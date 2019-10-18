import random
# generate a simple dataset with 3 labels
for i in range(200):
    x1 = random.uniform(0,10)
    x2 = random.uniform(0,6)
    x1 = round(x1,1)
    x2 = round(x2,1)
    label = ''
    if x1 < 5:
        label = 'A'
    elif (x1 >= 5) & (x2 >= 2):
        label = 'B'
    elif (x1 >= 5) & (x2 < 2):
        label = 'C'
    print(str(x1) + "," + str(x2) + "," + label)