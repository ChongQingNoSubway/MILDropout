import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F



class mildropout(nn.Module):
    def __init__(self,topk=3,kernel=7):
        super(mildropout,self).__init__()
        self.topk = topk
        self.kernel = kernel
        
    def forward(self,input):
        if not self.training:
            return input
        else:
            n,_ = input.shape
            #print(n)
            if n == 1:
                return input
            elif self.topk == 0:
                
                return input
            else:
                 
                importances = torch.mean(input,dim=1,keepdim=True)
                importances = torch.sigmoid(importances)

                mask = self.generate_mask(importances,input)
                input = input*mask

                
                ratio = (mask.numel() / mask.sum())
                input = input * ratio
                 

                return input
        
    def generate_mask(self,importance,input):
        n,f = input.shape
       
        mask = torch.zeros_like(importance)
        mask = mask.to(input.device)
        
        _, indx = torch.sort(importance,dim=0,descending=True)
        
        totall_idx = indx.view(-1)
        
        idx = totall_idx[:self.topk]
        
        remain_idx = totall_idx[self.topk:]

        top_k_feature = input[idx,:]

        top_remain_feature = input[remain_idx,:]


        x_1 = F.normalize(top_k_feature,dim=1,p=2)
        x_2 = F.normalize(top_remain_feature,dim=1,p=2)

        A = torch.mm(x_1,x_2.transpose(0,1))

        _,indx_A = torch.sort(A,dim=1,descending=True)

        delete_index = indx_A[:,:self.kernel]
        
        delete_index= torch.unique(delete_index)
        
        idx_remain = remain_idx[delete_index]
        

        totoall = torch.cat((idx_remain,idx))
        
        length = totoall.shape[0]

        source = torch.ones(length,1).cuda()
       
        mask.index_add_(0,totoall,source)
        
        mask = 1 - mask

        return mask
