import numpy as np
from skimage import io
from numpy import array
import matplotlib.pyplot as plt
#from typing import Dict

from src.common.registry import Registry
from src.metrics.base import Metric, GraphMetric
from src.common.utils import image_normalize, compute_pre_rec


@Registry.register_metric
class MAE(Metric):
    name: str = "mae"
    
    def compute(self,mask1,mask2):
        """
            Compute the mean absolute error
            
            Args:
                'mask1': HxW or HxWxn (asumme that all the n channels are the same and only the first channel will be used)
                'mask2': HxW or HxWxn
            
            Returns: 
                a value MAE, Mean Absolute Error
        """
        mask1 = array(mask1)
        mask2 = array(mask2)
        mask1 = image_normalize(mask1)
        mask2 = image_normalize(mask2)
        h,w = mask1.shape[0],mask1.shape[1]
        mask1 = image_normalize(mask1)
        mask2 = image_normalize(mask2)
        sumError = np.sum(np.absolute((mask1.astype(float) - mask2.astype(float))))
        maeError = sumError/((float(h)*float(w)+1e-8))

        return maeError
    

@Registry.register_metric    
class Precision(Metric):
    name: str = "precision"
    
    def compute(self,mask1,mask2):
        """
            Compute the precision
            
            Args:
                'mask1': ground truth
                'mask2': predictions
            
            Returns: 
                 precision
        """
        mask1 = array(mask1)
        mask2 = array(mask2)
        mask1 = image_normalize(mask1)
        mask2 = image_normalize(mask2)
        #true_values = mask1
        #predictions = mask2
        #print(true_values)
        #print(predictions)
        #N = true_values.shape[1]
        precision = (mask1==mask2).sum()/(mask2.sum()+1e-8)
        return precision
    
@Registry.register_metric    
class Recall(Metric):
    name: str="recall"
    
    def compute(self,mask1,mask2):
        """
            Compute the precision
            
            Args:
                'mask1': ground truth
                'mask2': predictions
            
            Returns: 
                 recall
        """
        mask1 = array(mask1)
        mask2 = array(mask2)
        mask1 = image_normalize(mask1)
        mask2 = image_normalize(mask2)
        recall = (mask1==mask2).sum()/(mask1.sum()+1e-8)
        return recall


@Registry.register_metric    
class F1(Metric):
    name: str="f1"
    
    def compute(self,mask1,mask2):
        """
            Compute the f1
            
            Args:
                'mask1': ground truth
                'mask2': predictions
            
            Returns: 
                 f1
        """

        mask1 = array(mask1)
        mask2 = array(mask2)
        mask1 = image_normalize(mask1)
        mask2 = image_normalize(mask2)
        precision = (mask1==mask2).sum()/(mask2.sum()+1e-8)
        recall = (mask1==mask2).sum()/(mask1.sum()+1e-8)
        return 2*recall*precision/(recall+precision)
