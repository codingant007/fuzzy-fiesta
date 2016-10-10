import sys
import re
from sets import Set
from PIL import Image
import numpy as np

WINDOW_SIZE = 32	# Size of the sample
HALF_WINDOW_SIZE = int(WINDOW_SIZE/2)

image_url = sys.argv[1]
KEY_URL = sys.argv[2]

image = Image.open(image_url)
f = open(KEY_URL)

no_of_ip = int(f.readline().split()[0])

def save_sample(image,x,y,sample_no):
	im_cropped = image.crop((x-HALF_WINDOW_SIZE,y-HALF_WINDOW_SIZE,x+HALF_WINDOW_SIZE,y+HALF_WINDOW_SIZE))
	sample_name = KEY_URL.split('.')[0]+ '-' + str(sample_no) +'.JPEG'
	im_cropped.save(sample_name)

i=0
for line in f.readlines():
	m = re.match(r'([0-9]*) ([0-9]*).*\n',line)
	if(m):
		x = int(float(m.group(1)))
		y = int(float(m.group(2)))
		i=i+1
		save_sample(image,x,y,i)
