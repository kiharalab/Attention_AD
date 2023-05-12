#
# Copyright (C) 2018 Xiao Wang
# Email:xiaowang20140001@gmail.com
#

#import parser
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-F',type=str, required=True,help='input path records sequences information')#File path for our MAINMAST code
    parser.add_argument('--mode',type=int,required=True,help='0: Predict for a sequence\n')
    parser.add_argument('-M',type=str,default="",help="model saving path")
    parser.add_argument('--gpu',type=str,default='0',help='gpu id choose for training')
    parser.add_argument('--class', type=int, default=2, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument("--seq_len",type=int,default=30,help="verify the number of elements in sequence")
    args = parser.parse_args()
    # try:
    #     import ray,socket
    #     rayinit()
    # except:
    #     print('ray need to be installed')#We do not need this since GAN can't be paralleled.
    params = vars(args)
    return params
