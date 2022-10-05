import numpy as np
from skimage import io

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
        if(len(mask1.shape)<2 or len(mask2.shape)<2):
            print("ERROR: Mask1 or mask2 is not matrix!")
            exit()
        if(len(mask1.shape)>2):
            mask1 = mask1[:,:,0]
        if(len(mask2.shape)>2):
            mask2 = mask2[:,:,0]
        if(mask1.shape!=mask2.shape):
            print("ERROR: The shapes of mask1 and mask2 are different!")
            exit()

        h,w = mask1.shape[0],mask1.shape[1]
        mask1 = image_normalize(mask1)
        mask2 = image_normalize(mask2)
        sumError = np.sum(np.absolute((mask1.astype(float) - mask2.astype(float))))
        maeError = sumError/(float(h)*float(w)+1e-8)

        return maeError
    
    
class AvgMae(Metric):
    name: str = "avg_mae"
    def compute(self,gt_name_list,rs_dir_lists):
        """
            Compute the average mean absolute error
            
            Args:
                'gt_name_list': ground truth name list
                'rs_dir_lists': to-be-evaluated mask directories (not the file names, just folder names)
            
            Returns: 
                average Mean Absolute Error, 1xN, N is the number of folders
                'gt2rs': numpy array with shape of (num_rs_dir)
        """

        num_gt = len(gt_name_list) # number of ground truth files
        num_rs_dir = len(rs_dir_lists) # number of method folders
        if(num_gt==0):
            print("ERROR: The ground truth directory is empty!")
            exit()

        mae = np.zeros((num_gt,num_rs_dir)) # MAE of methods
        gt2rs = np.zeros((num_gt,num_rs_dir)) # indicate if the mask mae of methods is correctly computed
        for i in range(0,num_gt):
            print('-Processed %d/%d'%(i+1,num_gt),end='\r')
            #print("Completed {:2.0%}".format(i / num_gt), end="\r") # print percentile of processed, python 3.0 and newer version
            gt = io.imread(gt_name_list[i]) # read ground truth
            gt_name = gt_name_list[i].split('/')[-1] # get the file name of the ground truth
            for j in range(0,num_rs_dir):
                tmp_mae = 0.0
                try:
                    rs = io.imread(rs_dir_lists[j]+gt_name) # read the corresponding mask of each method
                except IOError:
                    #print('ERROR: Couldn\'t find the following file:',rs_dir_lists[j]+gt_name)
                    continue
                try:
                    tmp_mae = MAE.compute(gt,rs) # compute the mae
                except IOError:
                    #print('ERROR: Fails in compute_mae!')
                    continue
                mae[i][j] = tmp_mae
                gt2rs[i][j] = 1.0
        mae_col_sum = np.sum(mae,0) # compute the sum of MAE of each method
        gt2rs = np.sum(gt2rs,0) # compute the number of correctly computed MAE of each method
        ave_maes = mae_col_sum/(gt2rs+1e-8) # compute the average MAE of each method
        return ave_maes, gt2rs
    
class Precision(Metric):
    name: str = "precision"
    
    def compute(self,gt_name_list,rs_dir_lists):
        """
            Compute the precision
            
            Args:
                'gt_name_list': ground truth name list
                'rs_dir_lists': to-be-evaluated mask directories (not the file names, just folder names)
            
            Returns: 
                 precision 'PRE': numpy array with shape of (num_rs_dir, 256)
                'gt2rs': numpy array with shape of (num_rs_dir)
        """

        mybins = np.arange(0,256) # different thresholds to achieve binarized masks for pre, rec, Fm measures

        num_gt = len(gt_name_list) # number of ground truth files
        num_rs_dir = len(rs_dir_lists) # number of method folders
        if(num_gt==0):
            #print("ERROR: The ground truth directory is empty!")
            exit()

        PRE = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # PRE: with shape of (num_gt, num_rs_dir, 256)
        REC = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # REC: the same shape with PRE
        # FM = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # Fm: the same shape with PRE
        gt2rs = np.zeros((num_gt,num_rs_dir)) # indicate if the mask of methods is correctly computed
        
        for i in range(0,num_gt):
            print('>>Processed %d/%d'%(i+1,num_gt),end='\r')
            gt = io.imread(gt_name_list[i]) # read ground truth
            gt = image_normalize(self,gt)*255.0 # convert gt to [0,255]
            gt_name = gt_name_list[i].split('/')[-1] # get the file name of the ground truth "xxx.png"

            for j in range(0,num_rs_dir):
                pre, rec, f = np.zeros(len(mybins)), np.zeros(len(mybins)), np.zeros(len(mybins)) # pre, rec, f or one mask w.r.t different thresholds
                try:
                    rs = io.imread(rs_dir_lists[j]+gt_name) # read the corresponding mask from each method
                    rs = image_normalize(rs)*255.0 # convert rs to [0,255]
                except IOError:
                    #print('ERROR: Couldn\'t find the following file:',rs_dir_lists[j]+gt_name)
                    continue
                try:
                    pre, rec = compute_pre_rec(gt,rs,mybins=np.arange(0,256))
                except IOError:
                    #print('ERROR: Fails in compute_mae!')
                    continue

                PRE[i,j,:] = pre
                REC[i,j,:] = rec
                gt2rs[i,j] = 1.0
        print('\n')
        gt2rs = np.sum(gt2rs,0) # num_rs_dir
        gt2rs = np.repeat(gt2rs[:, np.newaxis], 255, axis=1) #num_rs_dirx255

        PRE = np.sum(PRE,0)/(gt2rs+1e-8) # num_rs_dirx255, average PRE over the whole dataset at every threshold
        #REC = np.sum(REC,0)/(gt2rs+1e-8) # num_rs_dirx255
        #FM = (1+beta)*PRE*REC/(beta*PRE+REC+1e-8) # num_rs_dirx255
        
        return PRE, gt2rs
    
class Recall(Metric):
    name: str="recall"
    
    def compute(self,gt_name_list,rs_dir_lists):
        """
            Compute the recall
            
            Args:
                'gt_name_list': ground truth name list
                'rs_dir_lists': to-be-evaluated mask directories (not the file names, just folder names)
            
            Returns: 
                 recall 'REC': numpy array with shape of (num_rs_dir, 256)
                'gt2rs': numpy array with shape of (num_rs_dir)
        """

        mybins = np.arange(0,256) # different thresholds to achieve binarized masks for pre, rec, Fm measures

        num_gt = len(gt_name_list) # number of ground truth files
        num_rs_dir = len(rs_dir_lists) # number of method folders
        if(num_gt==0):
            #print("ERROR: The ground truth directory is empty!")
            exit()

        PRE = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # PRE: with shape of (num_gt, num_rs_dir, 256)
        REC = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # REC: the same shape with PRE
        # FM = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # Fm: the same shape with PRE
        gt2rs = np.zeros((num_gt,num_rs_dir)) # indicate if the mask of methods is correctly computed
        
        for i in range(0,num_gt):
            print('>>Processed %d/%d'%(i+1,num_gt),end='\r')
            gt = io.imread(gt_name_list[i]) # read ground truth
            gt = image_normalize(self,gt)*255.0 # convert gt to [0,255]
            gt_name = gt_name_list[i].split('/')[-1] # get the file name of the ground truth "xxx.png"

            for j in range(0,num_rs_dir):
                pre, rec, f = np.zeros(len(mybins)), np.zeros(len(mybins)), np.zeros(len(mybins)) # pre, rec, f or one mask w.r.t different thresholds
                try:
                    rs = io.imread(rs_dir_lists[j]+gt_name) # read the corresponding mask from each method
                    rs = image_normalize(rs)*255.0 # convert rs to [0,255]
                except IOError:
                    #print('ERROR: Couldn\'t find the following file:',rs_dir_lists[j]+gt_name)
                    continue
                try:
                    pre, rec = compute_pre_rec(gt,rs,mybins=np.arange(0,256))
                except IOError:
                    #print('ERROR: Fails in compute_mae!')
                    continue

                PRE[i,j,:] = pre
                REC[i,j,:] = rec
                gt2rs[i,j] = 1.0
        print('\n')
        gt2rs = np.sum(gt2rs,0) # num_rs_dir
        gt2rs = np.repeat(gt2rs[:, np.newaxis], 255, axis=1) #num_rs_dirx255

        #PRE = np.sum(PRE,0)/(gt2rs+1e-8) # num_rs_dirx255, average PRE over the whole dataset at every threshold
        REC = np.sum(REC,0)/(gt2rs+1e-8) # num_rs_dirx255
        #FM = (1+beta)*PRE*REC/(beta*PRE+REC+1e-8) # num_rs_dirx255
        
        return REC, gt2rs
    
class F1(Metric):
    name: str="f1"
    
    def compute(self,gt_name_list,rs_dir_lists,beta=0.3):
        """
            Compute the F1
            
            Args:
                'gt_name_list': ground truth name list
                'rs_dir_lists': to-be-evaluated mask directories (not the file names, just folder names)
            
            Returns: 
                 F1 'FM': numpy array with shape of (num_rs_dir, 256)
                'gt2rs': numpy array with shape of (num_rs_dir)
        """

        mybins = np.arange(0,256) # different thresholds to achieve binarized masks for pre, rec, Fm measures

        num_gt = len(gt_name_list) # number of ground truth files
        num_rs_dir = len(rs_dir_lists) # number of method folders
        if(num_gt==0):
            #print("ERROR: The ground truth directory is empty!")
            exit()

        PRE = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # PRE: with shape of (num_gt, num_rs_dir, 256)
        REC = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # REC: the same shape with PRE
        # FM = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # Fm: the same shape with PRE
        gt2rs = np.zeros((num_gt,num_rs_dir)) # indicate if the mask of methods is correctly computed
        
        for i in range(0,num_gt):
            print('>>Processed %d/%d'%(i+1,num_gt),end='\r')
            gt = io.imread(gt_name_list[i]) # read ground truth
            gt = image_normalize(self,gt)*255.0 # convert gt to [0,255]
            gt_name = gt_name_list[i].split('/')[-1] # get the file name of the ground truth "xxx.png"

            for j in range(0,num_rs_dir):
                pre, rec, f = np.zeros(len(mybins)), np.zeros(len(mybins)), np.zeros(len(mybins)) # pre, rec, f or one mask w.r.t different thresholds
                try:
                    rs = io.imread(rs_dir_lists[j]+gt_name) # read the corresponding mask from each method
                    rs = image_normalize(rs)*255.0 # convert rs to [0,255]
                except IOError:
                    #print('ERROR: Couldn\'t find the following file:',rs_dir_lists[j]+gt_name)
                    continue
                try:
                    pre, rec = compute_pre_rec(gt,rs,mybins=np.arange(0,256))
                except IOError:
                    #print('ERROR: Fails in compute_mae!')
                    continue

                PRE[i,j,:] = pre
                REC[i,j,:] = rec
                gt2rs[i,j] = 1.0
        print('\n')
        gt2rs = np.sum(gt2rs,0) # num_rs_dir
        gt2rs = np.repeat(gt2rs[:, np.newaxis], 255, axis=1) #num_rs_dirx255

        PRE = np.sum(PRE,0)/(gt2rs+1e-8) # num_rs_dirx255, average PRE over the whole dataset at every threshold
        REC = np.sum(REC,0)/(gt2rs+1e-8) # num_rs_dirx255
        FM = (1+beta)*PRE*REC/(beta*PRE+REC+1e-8) # num_rs_dirx255
        
        return FM, gt2rs
    
class PR_Curves(GraphMetric):
    name: str="precision-recall curve"
    def compute(self,PRE, REC, method_names, lineSylClr, linewidth, xrange=(0.0,1.0), yrange=(0.0,1.0), dataset_name = 'TEST', save_dir = './', save_fmt = 'jpg'):
        """
            obtain the precision recall curve
            
            Args:
                'PRE': precision numpy array
                'REC': recall numoy array
            
            Returns:
                None
        """
        fig1 = plt.figure(1)
        num = PRE.shape[0]
        for i in range(0,num):
            if (len(np.array(PRE[i]).shape)!=0):
                plt.plot(REC[i], PRE[i],lineSylClr[i],linewidth=linewidth[i],label=method_names[i])

        plt.xlim(xrange[0],xrange[1])
        plt.ylim(yrange[0],yrange[1])

        xyrange1 = np.arange(xrange[0],xrange[1]+0.01,0.1)
        xyrange2 = np.arange(yrange[0],yrange[1]+0.01,0.1)

        plt.tick_params(direction='in')
        plt.xticks(xyrange1,fontsize=15,fontname='serif')
        plt.yticks(xyrange2,fontsize=15,fontname='serif')

        ## draw dataset name
        plt.text((xrange[0]+xrange[1])/2.0,yrange[0]+0.02,dataset_name,horizontalalignment='center',fontsize=20, fontname='serif',fontweight='bold')

        plt.xlabel('Recall',fontsize=20,fontname='serif')
        plt.ylabel('Precision',fontsize=20,fontname='serif')

        font1 = {'family': 'serif',
                 'weight': 'bold',
                 'size': 10,
                 }

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [len(handles)-x for x in range(1,len(handles)+1)]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],loc='lower left', prop=font1)
        plt.grid(linestyle='--')
        fig1.savefig(save_dir+dataset_name+"_pr_curves."+save_fmt,bbox_inches='tight',dpi=300)
        print('>>PR-curves saved: %s'%(save_dir+dataset_name+"_pr_curves."+save_fmt))
        
    
class FM_Curves(GraphMetric):
    name: str="F-measure curve"
    def plot_save_fm_curves(self, FM, mybins, method_names, lineSylClr, linewidth, xrange=(0.0,1.0), yrange=(0.0,1.0), dataset_name = 'TEST', save_dir = './', save_fmt = 'pdf'):
        """
            obtain the precision recall curve
            
            Args:
                'F1': f-measure numpy array
            
            Returns:
                None
        """
        fig2 = plt.figure(2)
        num = FM.shape[0]
        for i in range(0,num):
            if (len(np.array(FM[i]).shape)!=0):
                plt.plot(np.array(mybins[0:-1]).astype(np.float)/255.0, FM[i],lineSylClr[i],linewidth=linewidth[i],label=method_names[i])

        plt.xlim(xrange[0],xrange[1])
        plt.ylim(yrange[0],yrange[1])

        xyrange1 = np.arange(xrange[0],xrange[1]+0.01,0.1)
        xyrange2 = np.arange(yrange[0],yrange[1]+0.01,0.1)

        plt.tick_params(direction='in')
        plt.xticks(xyrange1,fontsize=15,fontname='serif')
        plt.yticks(xyrange2,fontsize=15,fontname='serif')

        ## draw dataset name
        plt.text((xrange[0]+xrange[1])/2.0,yrange[0]+0.02,dataset_name,horizontalalignment='center',fontsize=20, fontname='serif',fontweight='bold')

        plt.xlabel('Thresholds',fontsize=20,fontname='serif')
        plt.ylabel('F-measure',fontsize=20,fontname='serif')

        font1 = {'family': 'serif',
                 'weight': 'bold',
                 'size': 10,
                 }

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [len(handles)-x for x in range(1,len(handles)+1)]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],loc='lower left', prop=font1)
        plt.grid(linestyle='--')
        fig2.savefig(save_dir+dataset_name+"_fm_curves."+save_fmt,bbox_inches='tight',dpi=300)
        print('>>F-measure curves saved: %s'%(save_dir+dataset_name+"_fm_curves."+save_fmt))
    
