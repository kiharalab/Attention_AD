from ops.utils import AverageMeter,Calculate_top_accuracy
import time
import numpy as np
import torch

def make_predictions(model,testloader,seq_length,encode_feature_len,params):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end_time = time.time()
    iteration = len(testloader)
    AA_Attention_Records = np.zeros([len(testloader.dataset), seq_length])
    Predict_Array = np.zeros(len(testloader.dataset))
    aa_feature_len = encode_feature_len + seq_length#
    for batch_idx, data in enumerate(testloader):
        aa_feature,index = data
        aa_feature = aa_feature.float()
        aa_feature = aa_feature.cuda()
        batch_size = aa_feature.size(0)
        h_0 = np.random.uniform(-0.5, high=0.5, size=(batch_size, aa_feature_len))
        c_0 = np.random.uniform(-0.5, high=0.5, size=(batch_size, aa_feature_len))
        h_0 = torch.from_numpy(h_0).float()
        c_0 = torch.from_numpy(c_0).float()
        h_0 = h_0.cuda()
        c_0 = c_0.cuda()
        data_time.update(time.time() - end_time, batch_size)
        with torch.no_grad():
            pred_batch,attention_batch = model(aa_feature,(h_0,c_0),use_attention=True)
        index = index.detach().cpu().numpy()
        attention_batch = attention_batch.detach().cpu().numpy()
        AA_Attention_Records[index] = attention_batch[:, 0, :]
        pred = torch.softmax(pred_batch, dim=1)
        pred = pred.detach().cpu().numpy()
        Predict_Array[index] = pred[:, 1]
        batch_time.update(time.time() - end_time, batch_size)
        end_time = time.time()

        print('Iter: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            .format(
            batch_idx + 1,
            iteration,
            batch_time=batch_time,
            data_time=data_time,))
    return Predict_Array,AA_Attention_Records