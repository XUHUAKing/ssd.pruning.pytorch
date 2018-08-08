# for XLab data
import os

fin = file('error.txt_test', 'r')
lines = fin.readlines()
keys = set()
for line in lines:
	line = line.strip()
	des = line.split(' ')
	path = des[0].split('.jpg')
	temp = path[0].split('/')
	tp = temp[len(temp)-2] + "/" + temp[len(temp)-1]
	keys.add(tp)
fin.close()

fin = file('test.txt', 'r')
fout = file('test_filter.txt', 'w')
lines = fin.readlines()
for line in lines:
	line = line.strip()
	if line not in keys:
		fout.write(line + "\n")
fin.close()
fout.close()
