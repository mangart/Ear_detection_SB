import cv2, sys, os

class Detector:
	# This example of a detector detects faces. However, you have annotations for ears!

	#cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'lbpcascade_frontalface.xml'))
	cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', "haarcascade_mcs_leftear.xml"))
	cascade1 = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', "haarcascade_mcs_rightear.xml"))

	def detect(self, img):
		det_list = self.cascade.detectMultiScale(img, 1.02, 1)
		det_list1 = self.cascade1.detectMultiScale(img, 1.02, 1)
		novi_seznam = list()
		novi_seznam.extend(det_list)
		novi_seznam.extend(det_list1)
		i = 0
		detectionList1 = []
		while i < len(novi_seznam):
			if novi_seznam[i][2] < 200 and novi_seznam[i][3] < 200:
				detectionList1.append(novi_seznam[i])
			i += 1
		return detectionList1
		#return novi_seznam
		#return det_list
		#return det_list1

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	detector = CascadeDetector()
	detected_loc = detector.detect(img)
	for x, y, w, h in detected_loc:
		cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
	cv2.imwrite(fname + '.detected.jpg', img)