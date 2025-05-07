# How Effective Can Dropout Be in Multiple Instance Learning? (ICML 2025) [![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/abs/2504.14783)

Paper link (preprint): [https://arxiv.org/abs/2504.14783]

## News :fire:
- **May 07, 2025:** Congratulations ! Paper has been accepted by ICML 2025 !

<img align="right" width="100%" height="100%" src="https://github.com/ChongQingNoSubway/MILDropout/blob/main/images/graphic_abstract.png">

> **Abstract.**  Multiple Instance Learning (MIL) is a popular weakly-supervised method for various applications, with a particular interest in histological whole slide image (WSI) classification. Due to the gigapixel resolution of WSI, applications of MIL in WSI typically necessitate a two-stage training scheme: first, extract features from the pre-trained backbone and then perform MIL aggregation. However, it is well-known that this suboptimal training scheme suffers from "noisy" feature embeddings from the backbone and inherent weak supervision, hindering MIL from learning rich and generalizable features. However, the most commonly used technique (i.e., dropout) for mitigating this issue has yet to be explored in MIL. In this paper, we empirically explore how effective the dropout can be in MIL. Interestingly, we observe that dropping the top-k most important instances within a bag leads to better performance and generalization even under noise attack. Based on this key observation, we propose a novel MIL-specific dropout method, termed MIL-Dropout, which systematically determines which instances to drop. Experiments on five MIL benchmark datasets and two WSI datasets demonstrate that MIL-Dropout boosts the performance of current MIL methods with a negligible computational cost.


## Key Code
```

import torch
import torch.nn as nn
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

```
## Applying Mildropout in MIL Models

When working with MIL (Multiple Instance Learning) models, it's common practice to apply **Mildropout** in the **shallow MLP layers**, particularly **after the ReLU activation function and before the data enters the MIL module**.

You can refer to the implementation in the following Python files for more details:
- `abmil.py`
- `dsmil.py`
- `transmil_mildropout.py`

### Example Usage

Here’s a simple example of how to apply Mildropout correctly:

```python
class TransMIL(nn.Module):
    def __init__(self, input_size, n_classes, mDim=512):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=mDim)
        self._fc1 = nn.Sequential(nn.Linear(input_size, mDim), nn.ReLU(), nn.Dropout(0.2))
        # the fc1 as the middle layer
        self._fc1 = nn.Sequential(
          nn.Linear(input_size, 256),
          nn.ReLU(),
          # append the mildropout after Relu activation function
          mildropout(topk=topk,kernel=kernel),
          nn.Linear(256, 128),
          nn.ReLU(),
          mildropout(topk=topk,kernel=kernel),
          nn.Linear(128, mDim),
          nn.ReLU(),
          mildropout(topk=topk,kernel=kernel),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=mDim)
        self.layer2 = TransLayer(dim=mDim)
        self.norm = nn.LayerNorm(mDim)
        self._fc2 = nn.Linear(mDim, self.n_classes)
```


## Wsi Pre-computed features and training/testing splits From DGR-MIL:
- DGR-MIL: https://github.com/ChongQingNoSubway/DGR-MIL

