# for XLab data
import os

fin = file('trainval.txt', 'r')
fout = file('trainval_0731.txt', 'w')
lines = fin.readlines()
for line in lines:
	line = line.strip()
	fout.write("/cephfs/share/data/VOC_xlab_products/JPEGImages/" + line +".jpg" + " " + "/cephfs/share/data/VOC_xlab_products/Annotations_24class/" + line + ".xml" + "\n")
fin.close()
fout.close()
