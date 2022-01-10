import cv2, sys
from skimage import feature
import numpy as np

class HOG:
	def __init__(self, resize=100, orientations=9):
		self.resize=resize
		self.orientations = orientations

	def extract(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))
		hog = feature.hog(img,self.orientations)
		
		return hog

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	extractor = Extractor()
	features = extractor.extract(img)
	print(features)