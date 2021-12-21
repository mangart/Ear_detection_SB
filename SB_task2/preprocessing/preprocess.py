import cv2
#from PIL import Image, ImageEnhance
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
import numpy as np

class Preprocess:

    def histogram_equlization_rgb(self, img):
        # Simple preprocessing using histogram equalization 
        # https://en.wikipedia.org/wiki/Histogram_equalization

        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)

        return img
    
    def edge_Enhance(self, img):
        # convert from cv2 image to pil image
        color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)
        # Apply edge enhancement filter
        edgeEnahnced = pil_image.filter(ImageFilter.EDGE_ENHANCE)
        # Apply increased edge enhancement filter
        #moreEdgeEnahnced = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        # use numpy to convert the pil_image into a numpy array
        numpy_image = np.array(edgeEnahnced)  

        # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that 
        # the color is converted from RGB to BGR format
        img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 
        return img
        
    def britness_correction(self, img, factor):
        color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)

        #image brightness enhancer
        enhancer = ImageEnhance.Brightness(pil_image)
        
        # apply brightness correction with a specified factor 
        im_output = enhancer.enhance(factor)
        
        # use numpy to convert the pil_image into a numpy array
        numpy_image = np.array(im_output)  

        # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that 
        # the color is converted from RGB to BGR format
        img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)         
        return img
        
    def gammaCorrection(self, img, gamma):
        invGamma = 1 / gamma
     
        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
     
        return cv2.LUT(img, table)
    # Add your own preprocessing techniques here.