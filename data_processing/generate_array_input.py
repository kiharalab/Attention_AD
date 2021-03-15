import os
import numpy as np


def generate_array_input(save_path,Seq_Map_Dict,train_data,seq_len):
    aa_feature_path = os.path.join(save_path,  "aa_feature.npy")
    AA_embed_len = len(Seq_Map_Dict)
    AA_feature = np.zeros([len(train_data), seq_len, AA_embed_len])

    for k in range(len(train_data)):
        aa_tmp=train_data[k]
        for j in range(seq_len):
            aa_code=aa_tmp[j]
            aa_index=Seq_Map_Dict[aa_code]
            AA_feature[k,j,aa_index]=1
    np.save(aa_feature_path, AA_feature)
    return aa_feature_path