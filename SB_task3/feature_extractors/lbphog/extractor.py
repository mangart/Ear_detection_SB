import cv2, sys
from skimage import feature
import numpy as np

class LBPHOG:
	def __init__(self, resize=100, num_points=24, radius=8, eps=1e-7):
		self.num_points = num_points * radius
		self.radius = radius
		self.eps = eps
		self.resize=resize

	def extract(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))

		lbp = feature.local_binary_pattern(img, self.num_points, self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, self.num_points + 3),range=(0, self.num_points + 2))
		hist = hist.astype("float")
		hist /= (hist.sum() + self.eps)
		hog = feature.hog(lbp,4)
		
		return hog

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	extractor = Extractor()
	features = extractor.extract(img)
	print(features)