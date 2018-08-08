# for XLab data
import os
import shutil
import cv2
import math
import xml.dom.minidom as suck
import sys
from sys import argv


cate_set = set()
mp = dict()


def work(jpg, xml, output_path):
	root = suck.parse(xml).documentElement
	objects = root.getElementsByTagName('object')
	img = cv2.imread(jpg)
	# print(jpg)

	for object in objects:
		item = object.getElementsByTagName('name')[0].childNodes[0].data
		xmin = math.floor(float(object.getElementsByTagName('xmin')[0].childNodes[0].data))
		xmax = math.ceil(float(object.getElementsByTagName('xmax')[0].childNodes[0].data))
		ymin = math.floor(float(object.getElementsByTagName('ymin')[0].childNodes[0].data))
		ymax = math.ceil(float(object.getElementsByTagName('ymax')[0].childNodes[0].data))

		cate_set.add(item)
		outfile = output_path + '/' + item + '/'
		if not os.path.exists(outfile):
			os.makedirs(outfile)

		if item not in mp:
			mp[item] = 0
		num = mp[item]
		mp[item] = num + 1

		source = xml.split('/')[-1].split('.')[0]
		name = source + '_' + item + '_' + str(num) + '.jpg'
		img2 = img[ymin:ymax, xmin:xmax]
		#cv2.imwrite(outfile + name, img2)
		# print(outfile + name)


if __name__ == "__main__":
	if len(argv) != 3:
		print('usage: python3 checkxml.py <table> <output_path>')
		exit()
	table, output_path = argv[1:]
	if output_path[-1] == '/':
		output_path = output_path[:-1]
	if not os.path.exists(output_path):
		print('output_path does not exist.')
		exit()
	for a in os.walk(output_path):
		e1, e2 = a[1:]
		if e1 or e2:
			print('output_path is not empty!')
			exit()

	with open(table, 'r', encoding='utf-8') as cin:
		A = cin.read().split('\n')

	ErrorJPG = []
	cnt = 0
	l = len(A)
	for a in A:
		if a == '':
			continue
		cnt += 1
		jpg, xml = a.split(' ')
		p = math.floor((cnt / l) * 100)
		sys.stdout.write('\r')
		sys.stdout.write('Working: ' + str(p) + '%')
		sys.stdout.flush()
		try:
			work(jpg, xml, output_path)
		except Exception as msg:
			ErrorJPG.append(a)
			# print(msg)

	sys.stdout.write('\n')
	sys.stdout.flush()

	with open('category.txt', 'w', encoding='utf-8') as cout:
		for cate in cate_set:
			cout.write(cate + '\n')
	with open('error.txt', 'w', encoding='utf-8') as cout:
		for error in ErrorJPG:
			cout.write(error + '\n')
	print('down.')
