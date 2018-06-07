import os 

fin = file('label2tag58.txt', 'r')
fout = file('label58.txt', 'w')
lines = fin.readlines()
for line in lines:
    line = line.strip()
    des = line.split('#')
    fout.write(des[0]+"\n")
fin.close()
fout.close()
