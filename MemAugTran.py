import torch.nn as nn
import torch
import math
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim=512,num_heads=8):
        super(MultiHeadAttention,self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.wq = nn.Linear(embed_dim,embed_dim)
        self.wk = nn.Linear(embed_dim,embed_dim)
        self.wv = nn.Linear(embed_dim,embed_dim)

        self.wo = nn.Linear(embed_dim,embed_dim)

        self.dwconv1 = nn.Conv1d(in_channels=embed_dim,out_channels=embed_dim,kernel_size=3,padding=1,groups=embed_dim)
        self.dwconv2 = nn.Conv1d(in_channels=embed_dim,out_channels=embed_dim,kernel_size=3,padding=1,groups=embed_dim)
        self.dwconv3 = nn.Conv1d(in_channels=embed_dim,out_channels=embed_dim,kernel_size=3,padding=1,groups=embed_dim)


    def forward(self,x):
        """
        :param x:[b,64,512]
        :return:
        """
        b,seq,embed = x.shape

        query = self.dwconv1(self.wq(x).transpose(1,2)).transpose(1,2)
        key = self.dwconv2(self.wk(x).transpose(1,2)).transpose(1,2)
        value = self.dwconv3(self.wv(x).transpose(1,2)).transpose(1,2)

        query = query.view(b,seq,self.num_heads,self.head_dim)
        query = query.transpose(1,2)

        key = key.view(b,seq,self.num_heads,self.head_dim).transpose(1,2)

        value = value.view(b,seq,self.num_heads,self.head_dim).transpose(1,2)

        scores = query@key.transpose(-2,-1)
        scores = scores / math.sqrt(self.head_dim)
        attn = torch.softmax(scores,dim=-1)

        out = attn @ value
        out = out.transpose(1,2).reshape(b,seq,-1)

        out = self.wo(out)
        return out

class Memory(nn.Module):
    def __init__(self,mem_dim=2000,feature_dim=512,shrink_thres=0.0025):
        super(Memory,self).__init__()
        self.mem_dim = mem_dim
        self.feature_dim = feature_dim
        self.shrink = shrink_thres

        self.memory_matrix = nn.Parameter(torch.FloatTensor(mem_dim,feature_dim))
        stdv = 1./ math.sqrt(feature_dim)
        self.memory_matrix.data.uniform_(-stdv,stdv)

    def forward(self,x):
        """
        :param x:[b,512,64]
        """
        b,c,hw = x.shape

        x_reshape = x.permute(0,2,1).contiguous()
        x_norm = F.normalize(x_reshape,dim=2)
        mem_norm = F.normalize(self.memory_matrix,dim=1)
        attn_weight = torch.matmul(x_norm,mem_norm.transpose(0,1))

        scale = 10.0
        attn_weight_soft = F.softmax(attn_weight*scale,dim=2)
        attn_weight_hard = F.relu(attn_weight_soft-self.shrink)
        ctx_sum = torch.sum(attn_weight_hard,dim=2,keepdim=True)
        attn_weight_hard = attn_weight_hard / (ctx_sum + 1e-12)

        refuse_feature = torch.matmul(attn_weight,self.memory_matrix)
        refuse_feature = refuse_feature.permute(0,2,1).contiguous()

        return refuse_feature,attn_weight_hard,attn_weight_soft

class MemoryAugmentedMHA(nn.Module):
    def __init__(self,embed_dim=512,num_head=8,mem_dim=1000,shrink_thres=0.0025):
        super(MemoryAugmentedMHA,self).__init__()
        self.mha = MultiHeadAttention(embed_dim=embed_dim,num_heads=num_head)
        self.memory = Memory(mem_dim=mem_dim,feature_dim=512,shrink_thres=shrink_thres)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        """
        :param x:[b,512,64]
        :return:
        """
        x_permute = x.permute(0,2,1)
        x_mha = self.mha(x_permute)
        x_mha = x_mha.permute(0,2,1)
        x_fused = x + self.gamma*x_mha
        x_recon,attn_weight_hard,attn_weight_soft = self.memory(x_fused)
        return x_recon,attn_weight_soft
