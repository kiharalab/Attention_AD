# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import torch
import torch.utils.data as data
import numpy as np
import random
import os

class AD_Dataset(data.Dataset):
    def __init__(self, feature_path,seq_len,params):
        super(AD_Dataset, self).__init__()
        self.AA_Feature=np.load(feature_path)
        self.position_array = self.Build_Order_Matrix(seq_len)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        aa_feature=self.AA_Feature[index]#seq_len*embeding_length
        aa_feature = np.concatenate([aa_feature, self.position_array], axis=1)
        return aa_feature,index
    def Build_Order_Matrix(self,seq_len):
        array=np.zeros([seq_len,seq_len])
        for k in range(seq_len):
            array[k,k]=1
        return array

    def __len__(self):
        return len(self.AA_Feature)


