
import os
from ops.os_operation import mkdir
from data_processing.read_input import read_input,verify_input
from model.Attention_LSTM import Attention_LSTM
from data_processing.generate_array_input import generate_array_input
from data_processing.AD_Dataset import AD_Dataset
from evaluate.make_predictions import make_predictions

import torch
import torch.nn as nn
import json

def init_save_path(file_name):
    save_path=os.path.join(os.getcwd(),"predict_result")
    mkdir(save_path)
    save_path=os.path.join(save_path,file_name)
    mkdir(save_path)
    return save_path


def prepare_dataLoader(feature_path,params):

    Testing_Dataset = AD_Dataset(feature_path,params['seq_len'],params)
    test_dataloader = torch.utils.data.DataLoader(Testing_Dataset, batch_size=params['batch_size'],
                                                 shuffle=False, num_workers=int(params['num_workers']),
                                                 drop_last=False, pin_memory=True)
    return test_dataloader

def predict_seq(input_path,model_path,seq_len,num_classes,params):
    input_list=read_input(input_path)
    #verify input
    input_list=verify_input(input_list,seq_len)
    model=Attention_LSTM(seq_len,num_classes)

    #load model state dict
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    state = torch.load(model_path)
    msg=model.load_state_dict(state['state_dict'], strict=False)
    print("loading msg:",msg)
    model.eval()

    #prepare save path
    split_file_name=os.path.split(input_path)[1]
    save_path=init_save_path(split_file_name)
    #create dataset here
    #get id dict to build connections from
    aa_id_path=os.path.join(os.getcwd(),"data_processing")
    aa_id_path=os.path.join(aa_id_path,"aa_id.txt")
    with open(aa_id_path, 'r') as file:
        aa_map_Dict=json.load(file)
    feature_path=generate_array_input(save_path,aa_map_Dict,input_list,seq_len)
    #specify in the dataset
    testloader=prepare_dataLoader(feature_path,params)
    #get predictions
    Predict_Array,Attention_Array=make_predictions(model,testloader,seq_len,len(aa_map_Dict),params)
    output_path=os.path.join(save_path,"Report.txt")
    with open(output_path,'w') as file:
        file.write("Sequence\tAttention\tProb\tLabel")
        for k in range(len(input_list)):
            input_seq=input_list[k]
            predict_label=Predict_Array[k]
            attention_info=Attention_Array[k]
            file.write(input_seq+"\t")
            for j in range(len(attention_info)):
                file.write("%.6f,"%attention_info[j])
            file.write("\t")
            file.write("%.4f\t"%predict_label)
            if predict_label>0.5:
                file.write("1 ")
            else:
                file.write("0 ")
            file.write("\n")







