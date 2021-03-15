


import torch
import torch.nn as nn
import math
import numpy as np

class Attention_LSTM(nn.Module):

    def __init__(self,aa_feature_len, output_class):
        """

        :param aa_feature_len: amino acid feature embedding length
        :param output_class: number of final output classes
        """
        super(Attention_LSTM, self).__init__()

        self.aa_fc=nn.Linear(aa_feature_len, aa_feature_len)
        self.fc2 = nn.Linear(aa_feature_len , output_class)
        self.aa_lstm= nn.LSTMCell(aa_feature_len, aa_feature_len)
    def forward_aaonly(self, aa_feature,aa_hidden,use_attention=False):

        #ss_feature: batch_size*seq_len*feature_size

        aa_h0,aa_c0=aa_hidden
        seq_len=aa_feature.size(1)
        aa_new_feature=[]
        for k in range(seq_len):
            # embed feature
            aa_feature_tmp = self.aa_fc(aa_feature[:, k, :])
            aa_new_feature.append(aa_feature_tmp)

        aa_new_feature = torch.stack(aa_new_feature, dim=1)
        aa_feature=aa_new_feature
        #ss_feature0=ss_feature.detach()
        #aa_feature0=aa_feature.detach()
        for k in range(seq_len):
            cur_aa=aa_feature[:,k,:]
            aa_h0, aa_c0 = self.aa_lstm(cur_aa, (aa_h0, aa_c0))
        #calculate attention matrix
        aa_weight_matrix = torch.bmm(aa_h0.view(aa_h0.size(0), 1, -1),
                                     aa_feature.permute(0, 2, 1))  # batch_size*1*seq_length

        aa_weight_matrix = torch.softmax(aa_weight_matrix, dim=2)
        sum_aa_combine = torch.bmm(aa_weight_matrix, aa_feature).squeeze(1)  # batch_size*feature_len
        sum_feature=sum_aa_combine

        x=self.fc2(sum_feature)
        if use_attention:
            return x,aa_weight_matrix

        return x


    def forward(self, aa_feature,aa_hidden,use_attention=False):
        """
        :param aa_feature:
        :param aa_hidden:
        :param use_attention: return attention matrix or not
        :return:
        """
        x = self.forward_aaonly(aa_feature, aa_hidden, use_attention)
        return x
