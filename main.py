

from ops.argparser import argparser
import os
if __name__ == "__main__":
    params = argparser()
    #print(params)
    if params['mode']==0:
        #give a path of txt files, each line is the sequence
        input_path=os.path.abspath(params['F'])
        model_path=os.path.abspath(params['M'])
        seq_length=params['seq_len']
        choose = params['gpu']
        os.environ["CUDA_VISIBLE_DEVICES"] = choose
        num_classes=params['class']
        from evaluate.predict_seq import predict_seq
        predict_seq(input_path,model_path,seq_length,num_classes,params)
