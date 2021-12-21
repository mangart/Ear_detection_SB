import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def get_annotations(self, annot_name):
            with open(annot_name) as f:
                lines = f.readlines()
                annot = []
                for line in lines:
                    l_arr = line.split(" ")[1:5]
                    l_arr = [int(i) for i in l_arr]
                    annot.append(l_arr)
            return annot

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        iou_arr = []
        mAP_arr = []
        TP = 0
        preprocess = Preprocess()
        eval = Evaluation()
        
        # Change the following detector and/or add your detectors below
        import detectors.cascade_detector.detector as cascade_detector
        # import detectors.your_super_detector.detector as super_detector
        cascade_detector = cascade_detector.Detector()
        
        
        for im_name in im_list:
            
            # Read an image
            img = cv2.imread(im_name)

            # Apply some preprocessing
            #img = preprocess.histogram_equlization_rgb(img) # This one makes VJ worse
            #tryed edge enhancement preprossesing and the iou score droped from 33.21% to 19.6% and mAP score droped from 19.93% to 15.6%
            #img = preprocess.edge_Enhance(img)
            # tryed britness correction and a slighter darker image gives slightly better results iou 33.51% and mAP 20.80%
            img = preprocess.britness_correction(img,0.9)
            
            # Run the detector. It runs a list of all the detected bounding-boxes. In segmentor you only get a mask matrices, but use the iou_compute in the same way.
            prediction_list = cascade_detector.detect(img)

            # Read annotations:
            annot_name = os.path.join(self.annotations_path, Path(os.path.basename(im_name)).stem) + '.txt'
            annot_list = self.get_annotations(annot_name)

            # Only for detection:
            p, gt = eval.prepare_for_detection(prediction_list, annot_list)
            
            iou = eval.iou_compute(p, gt)
            mAP = eval.mAP_compute(p,gt)
            if mAP == 1:
                TP += 1
            iou_arr.append(iou)
            mAP_arr.append(mAP)
        miou = np.average(iou_arr)
        mmAP = TP / len(mAP_arr)
        print("\n")
        print("Average IOU:", f"{miou:.2%}")
        print("\n")
        print("Average mAP:", f"{mmAP:.2%}")
        print("\n")


if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()