import os

# filter "/cephfs/group files"
fin = open('train_58_0713.txt', 'r')
fout = open('train_58_0713_filter.txt', 'w') # open for python3, file for python2
lines = fin.readlines()
keys = set()
for line in lines:
    line = line.strip()
    des = line.split(' ')
    path = des[0].split('.jpg')
    temp = path[0].split('/')
    if temp[2] == 'group':
        print(line)
        continue
    fout.write(line+"\n")
fin.close()
fout.close()
