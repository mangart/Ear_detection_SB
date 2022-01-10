import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist 
from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition import Evaluation

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config_recognition.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def clean_file_name(self, fname):
        return fname.split('/')[1].split(' ')[0]
    
    # vrne slovar z imenom slike in vrednostjo razred oziroma kateri osebi pripada uho primer 'train/0001.png': 100
    def get_annotations(self, annot_f):
        d = {}
        with open(annot_f) as f:
            lines = f.readlines()
            for line in lines:
                (key, val) = line.split(',')
                # keynum = int(self.clean_file_name(key))
                d[key] = int(val)
        return d

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        i = 0
        while i < len(im_list):
            im_list[i] = im_list[i][:33] + '/0' + im_list[i][34 + 1:]
            i += 1
        #print(im_list)
        iou_arr = []
        preprocess = Preprocess()
        eval = Evaluation()
        resize = 100
        orientations = 9
        cla_d = self.get_annotations(self.annotations_path)
        #print(cla_d)
        # Change the following extractors, modify and add your own

        # Pixel-wise comparison:
        import feature_extractors.pix2pix.extractor as p2p_ext
        pix2pix = p2p_ext.Pix2Pix(resize)
        
        #lbp extractor
        num_points = 8
        radius = 3
        import feature_extractors.lbp.extractor as lbp_ext
        lbp = lbp_ext.LBP(resize,num_points,radius,1e-7)
        
        #hog extractor
        import feature_extractors.hog.extractor as hog_ext
        hog = hog_ext.HOG(resize,orientations)
        
        #lbphog extractor
        import feature_extractors.lbphog.extractor as lbphog_ext
        lbphog = lbphog_ext.LBPHOG(resize)
        
        lbp_features_arr = []
        plain_features_arr = []
        hog_features_arr = []
        lbphog_features_arr = []
        y = []

        for im_name in im_list:
            
            # Read an image
            img = cv2.imread(im_name)

            y.append(cla_d['/'.join(im_name.split('/')[-2:])])
            # Apply some preprocessing here
            img = preprocess.britness_correction(img,0.4) #0.3 0.7 0.9
            #img = preprocess.edge_Enhance(img)
            # Run the feature extractors            
            plain_features = pix2pix.extract(img)
            plain_features_arr.append(plain_features)
            
            lbp_features = lbp.extract(img)
            lbp_features_arr.append(lbp_features)
            
            hog_features = hog.extract(img)
            hog_features_arr.append(hog_features)
            
            lbphog_features = lbphog.extract(img)
            lbphog_features_arr.append(lbphog_features)

        mera_razdalje = 'correlation' # jensenshannon correlation cosine kulsinski russellrao
        Y_plain = cdist(plain_features_arr, plain_features_arr, mera_razdalje)
        Y_plain_lbp = cdist(lbp_features_arr, lbp_features_arr, mera_razdalje)
        Y_plain_hog = cdist(hog_features_arr, hog_features_arr, mera_razdalje)
        Y_plain_lbphog = cdist(lbphog_features_arr, lbphog_features_arr, mera_razdalje)
        #print(Y_plain)
        #print(y)
        #r1 = eval.compute_rank1(Y_plain, y)
        #print('Pix2Pix Rank-1[%]', r1)
        #r1_lbp = eval.compute_rank1(Y_plain_lbp, y)
        #print('LBP Rank-1[%]', r1_lbp)
        #r1_hog = eval.compute_rank1(Y_plain_hog, y)
        #print('HOG Rank-1[%]', r1_hog)
        #r1_lbphog = eval.compute_rank1(Y_plain_lbphog, y)
        #print('LBPHOG Rank-1[%]', r1_lbphog)
        
        razredi_po_slikah_pix = eval.rankc_for_plot(Y_plain,y)
        razredi_po_slikah_lbp = eval.rankc_for_plot(Y_plain_lbp,y)
        razredi_po_slikah_hog = eval.rankc_for_plot(Y_plain_hog,y)
        razredi_po_slikah_lbphog = eval.rankc_for_plot(Y_plain_lbphog,y)
        print('Rank-1')
        r1_pix = eval.calc_rankc(razredi_po_slikah_pix,y,1)
        print('Pix2Pix Rank-1[%]', r1_pix)        
        r1_lbp1 = eval.calc_rankc(razredi_po_slikah_lbp, y,1)
        print('LBP Rank-1[%]', r1_lbp1)    
        r1_hog1 = eval.calc_rankc(razredi_po_slikah_hog, y,1)
        print('HOG Rank-1[%]', r1_hog1) 
        r1_lbphog1 = eval.calc_rankc(razredi_po_slikah_lbphog, y,1)
        print('LBPHOG Rank-1[%]', r1_lbphog1)
        print('Rank-5')
        r5_pix = eval.calc_rankc(razredi_po_slikah_pix,y,5)
        print('Pix2Pix Rank-5[%]', r5_pix)        
        r5_lbp1 = eval.calc_rankc(razredi_po_slikah_lbp, y,5)
        print('LBP Rank-5[%]', r5_lbp1)    
        r5_hog1 = eval.calc_rankc(razredi_po_slikah_hog, y,5)
        print('HOG Rank-5[%]', r5_hog1) 
        r5_lbphog1 = eval.calc_rankc(razredi_po_slikah_lbphog, y,5)
        print('LBPHOG Rank-5[%]', r5_lbphog1)
        print('Rank-10')
        r10_pix = eval.calc_rankc(razredi_po_slikah_pix,y,10)
        print('Pix2Pix Rank-10[%]', r10_pix)        
        r10_lbp1 = eval.calc_rankc(razredi_po_slikah_lbp, y,10)
        print('LBP Rank-10[%]', r10_lbp1)    
        r10_hog1 = eval.calc_rankc(razredi_po_slikah_hog, y,10)
        print('HOG Rank-10[%]', r10_hog1) 
        r10_lbphog1 = eval.calc_rankc(razredi_po_slikah_lbphog, y,10)
        print('LBPHOG Rank-10[%]', r10_lbphog1)
        print('Rank-20')
        r20_pix = eval.calc_rankc(razredi_po_slikah_pix,y,20)
        print('Pix2Pix Rank-20[%]', r20_pix)        
        r20_lbp1 = eval.calc_rankc(razredi_po_slikah_lbp, y,20)
        print('LBP Rank-20[%]', r20_lbp1)    
        r20_hog1 = eval.calc_rankc(razredi_po_slikah_hog, y,20)
        print('HOG Rank-20[%]', r20_hog1) 
        r20_lbphog1 = eval.calc_rankc(razredi_po_slikah_lbphog, y,20)
        print('LBPHOG Rank-20[%]', r20_lbphog1)        
        eval.CMC_plot(Y_plain,y,'graf_pix.png')
        eval.CMC_plot(Y_plain_lbp,y,'graf_lbp.png')
        eval.CMC_plot(Y_plain_hog,y,'graf_hog.png')
        eval.CMC_plot(Y_plain_lbphog,y,'graf_lbphog.png')
        

if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()