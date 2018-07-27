import os
import shutil
import cv2
cv2.setNumThreads(0) # pytorch issue 1355: possible deadlock in DataLoader
# OpenCL may be enabled by default in OpenCV3;
# disable it because it because it's not thread safe and causes unwanted GPU memory allocations
cv2.ocl.setUseOpenCL(False)
import math
import xml.dom.minidom as suck
import sys
from sys import argv

cate_set = set()
mp = dict()


def check(jpg, xml, output_path):
	root = suck.parse(xml).documentElement
	xml_size = root.getElementsByTagName('size')[0]
	xml_size = (int(xml_size.getElementsByTagName('height')[0].childNodes[0].data),
				int(xml_size.getElementsByTagName('width')[0].childNodes[0].data),
				int(xml_size.getElementsByTagName('depth')[0].childNodes[0].data))
	objects = root.getElementsByTagName('object')
	img = cv2.imread(jpg)
	img_size = img.shape
	flag = False
	if xml_size != img_size:
		raise Exception('@shape_is_not_right')

	for object in objects:
		item = object.getElementsByTagName('name')[0].childNodes[0].data
		xmin = math.floor(float(object.getElementsByTagName('xmin')[0].childNodes[0].data))
		xmax = math.ceil(float(object.getElementsByTagName('xmax')[0].childNodes[0].data))
		ymin = math.floor(float(object.getElementsByTagName('ymin')[0].childNodes[0].data))
		ymax = math.ceil(float(object.getElementsByTagName('ymax')[0].childNodes[0].data))

		if xmin < 0 or ymin < 0 or xmax > xml_size[1] or ymax > xml_size[0] or xmax < xmin or ymax < ymin:
			raise Exception('@object_out_of_scope')

		cate_set.add(item)
		if flag == False:
			with open('checkxml_list.txt', 'a', encoding='utf-8') as outer:
				outer.write(jpg +" "+xml + '\n')
		flag = True

'''
		outfile = os.path.join(output_path, item)
		if not os.path.exists(outfile):
			os.makedirs(outfile)

		if item not in mp:
			mp[item] = 0
		num = mp[item]
		mp[item] = num + 1

		source = '.'.join(xml.split('/')[-1].split('.')[:-1])
		name = '_'.join([source, item, str(num)]) + '.jpg'
		img2 = img[ymin:ymax, xmin:xmax]
		cv2.imwrite(os.path.join(outfile, name), img2)
'''

def check_argv():
	if len(argv) != 3:
		exit('usage: python3 checkxml.py <jpg_xml_list> <output_path>')
	jpg_xml_list_file, output_path = argv[1:]
	if not os.path.exists(output_path):
		exit('output_path does not exist.')
	if os.listdir(output_path):
		exit('output_path is not empty!')
	return jpg_xml_list_file, output_path


def work(jpg_xml_list, output_path):
	ErrorJPG = []
	cnt = 0
	l = len(jpg_xml_list)
	for jpg_xml in jpg_xml_list:
		if jpg_xml == '':
			continue
		cnt += 1
		jpg, xml = jpg_xml.split(' ')
		sys.stdout.write('\r')
		sys.stdout.write('Working: ' + str(math.floor((cnt / l) * 100)) + '%')
		sys.stdout.flush()
		try:
			check(jpg, xml, output_path)
		except Exception as msg:
			msg = str(msg)
			if msg[0] == '@':
				msg = msg[1:]
			else:
				msg = 'unknown'
			ErrorJPG.append(jpg_xml + ' ' + msg)
	sys.stdout.write('\n')
	sys.stdout.flush()
	return ErrorJPG


def _input(jpg_xml_list_file):
	with open(jpg_xml_list_file, 'r', encoding='utf-8') as jpg_xml_list_reader:
		return jpg_xml_list_reader.read().split('\n')


def _output(ErrorJPG):
	with open('checkxml_category.txt', 'w', encoding='utf-8') as category_outer:
		for cate in cate_set:
			category_outer.write(cate + '\n')
	with open('checkxml_error.txt', 'w', encoding='utf-8') as error_outer:
		for error in ErrorJPG:
			error_outer.write(error + '\n')


def main():
	jpg_xml_list_file, output_path = check_argv()
	jpg_xml_list = _input(jpg_xml_list_file)
	ErrorJPG = work(jpg_xml_list, output_path)
	_output(ErrorJPG)
	print('down.')


if __name__ == "__main__":
	main()
