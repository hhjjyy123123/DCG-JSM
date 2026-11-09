import torch
import torch.nn as nn
from model.basic import ResBlock_V3,ResNeXtBlockFixed,UpSample,GatedCBAM_V2
from model.MemAugTran import MemoryAugmentedMHA
from model.DynamicEmbedding import ConvGRU


class Encoder(nn.Module):
    def __init__(self,in_ch=12,feature_root=16,out_ch=512):
        super(Encoder,self).__init__()
        """
        input:[b,12,256,256]
        output:[b,512,8,8]
        """
        self.normconv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=feature_root,kernel_size=1,stride=1,padding=0),
            nn.GroupNorm(num_groups=1,num_channels=feature_root),
            nn.ReLU(inplace=False)
        )

        self.res1 = ResBlock_V3(in_ch=feature_root,out_ch=feature_root*2,num_group=2,reduction_rate=4)
        self.res2 = ResBlock_V3(in_ch=feature_root*2,out_ch=feature_root*4,num_group=2,reduction_rate=4)
        self.res3 = ResBlock_V3(in_ch=feature_root*4,out_ch=feature_root*8,num_group=4,reduction_rate=8)
        self.res4 = ResBlock_V3(in_ch=feature_root*8,out_ch=feature_root*16,num_group=8,reduction_rate=16)
        self.res5 = ResBlock_V3(in_ch=feature_root*16,out_ch=out_ch,num_group=8,reduction_rate=16)

    def forward(self,input_tensor):
        input_tensor = self.normconv(input_tensor)  #[b,12,256,256] => [b,16,256,256]
        x1 = self.res1(input_tensor)     #[b,16,256,256] => [b,32,128,128]
        x2 = self.res2(x1)               #[b,32,128,128] => [b,64,64,64]
        x3 = self.res3(x2)               #[b,64,64,64] => [b,128,32,32]
        x4 = self.res4(x3)               #[b,128,32,32] => [b,256,16,16]
        x5 = self.res5(x4)               #[b,256,16,16] => [b,512,8,8]
        return x5,x4,x3,x2,x1


class Decoder(nn.Module):
    def __init__(self,in_ch=512,out_ch=3,feature_root=16):
        super(Decoder,self).__init__()

        self.upsample1 = UpSample(in_ch=in_ch,out_ch=feature_root*16)
        self.resfix1 = ResNeXtBlockFixed(channels=feature_root*16*2,division=32,num_group=16)
        self.gcbam1 = GatedCBAM_V2(in_ch=feature_root*16,reduction=8,kernel_size=5)

        self.upsample2 = UpSample(in_ch=feature_root*16*2,out_ch=feature_root*8)
        self.resfix2 = ResNeXtBlockFixed(channels=feature_root*8*2,division=16,num_group=8)
        self.gcbam2 = GatedCBAM_V2(in_ch=feature_root*8,reduction=8,kernel_size=5)

        self.upsample3 = UpSample(in_ch=feature_root*8*2,out_ch=feature_root*4)
        self.resfix3 = ResNeXtBlockFixed(channels=feature_root*4*2,division=16,num_group=8)
        self.gcbam3 = GatedCBAM_V2(in_ch=feature_root*4,reduction=4,kernel_size=7)

        self.upsample4 = UpSample(in_ch=feature_root*4*2,out_ch=feature_root*2)
        self.resfix4 = ResNeXtBlockFixed(channels=feature_root*2*2,division=8,num_group=4)
        self.gcbam4 = GatedCBAM_V2(in_ch=feature_root*2,reduction=4,kernel_size=7)

        self.upsample5 = UpSample(in_ch=feature_root*2*2,out_ch=feature_root)
        self.resfix5 = ResNeXtBlockFixed(channels=feature_root,division=2,num_group=2)

        self.Gen = nn.Sequential(
            nn.Conv2d(in_channels=feature_root, out_channels=feature_root // 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=feature_root // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=feature_root // 2,out_channels=out_ch,kernel_size=3,padding=1,stride=1),
            nn.Tanh()
        )



    def forward(self,x,tensor4,tensor3,tensor2,tensor1):
        x1 = self.upsample1(x) #[b,512,8,8] => [b,256,16,16]
        tensor4_filtered = self.gcbam1.forward(f_encoder=tensor4,f_decoder=x1)
        x1_concat = torch.concat([x1,tensor4_filtered],dim=1)     #[b,512,16,16]
        x1_concat = self.resfix1(x1_concat)                        #[b,512,16,16]

        x2 = self.upsample2(x1_concat)   #[b,512,16,16] => [b,128,32,32]
        tensor3_filtered = self.gcbam2.forward(f_encoder=tensor3,f_decoder=x2)
        x2_concat = torch.concat([x2,tensor3_filtered],dim=1)  #[b,256,32,32]
        x2_concat = self.resfix2(x2_concat)

        x3 = self.upsample3(x2_concat)   #[b,128,32,32] => [b,64,64,64]
        tensor2_filtered = self.gcbam3.forward(f_encoder=tensor2,f_decoder=x3)
        x3_concat = torch.concat([x3,tensor2_filtered],dim=1)    #[b,128,64,64]
        x3_concat = self.resfix3(x3_concat)


        x4 = self.upsample4(x3_concat)   #[b,128,64,64] => [b,32,128,128]
        tensor1_filtered = self.gcbam4.forward(f_encoder=tensor1,f_decoder=x4)
        x4_concat = torch.concat([x4,tensor1_filtered],dim=1)   #[b,64,128,128]
        x4_concat = self.resfix4(x4_concat)

        x5 = self.upsample5(x4_concat)   #[b,64,128,128] => [b,16,256,256]
        x5 = self.resfix5(x5)

        out = self.Gen(x5)
        return out,tensor1_filtered


class Generator(nn.Module):
    def __init__(self,in_ch=12,out_ch=3,feature_root=16,mem_dim=1000,shrink_thres=0.0025,num_layers=4):
        super(Generator,self).__init__()

        self.encoder = Encoder(in_ch=in_ch,out_ch=feature_root*32,feature_root=feature_root)
        self.decoder = Decoder(in_ch=feature_root*32,out_ch=out_ch,feature_root=feature_root)

        self.gru = ConvGRU(in_ch=feature_root*32,hidden_ch=128,num_group=8,num_frame=4,num_layers=num_layers)

        self.merge = nn.Conv2d(in_channels=feature_root*32*2,out_channels=feature_root*32,kernel_size=1)

        self.bottle = MemoryAugmentedMHA(embed_dim=512, mem_dim=mem_dim, num_head=8, shrink_thres=shrink_thres)

    def forward(self,input_x):
        encode_tensor,tensor4,tensor3,tensor2,tensor1 = self.encoder(input_x)  #[b,512,8,8]
        b,c,th,tw = encode_tensor.shape
        encode_tensor_reshape = encode_tensor.view(b,c,th*tw)  #[b,512,64]
        att_encode,attn_weight = self.bottle(encode_tensor_reshape)
        att_encode = att_encode.view(b,c,th,tw)    #[b,512,8,8]
        encode_tensor_dynamic = self.gru(encode_tensor) #[b,512,8,8]
        feature_dec = self.merge(torch.concat([att_encode,encode_tensor_dynamic],dim=1))
        #对编码器的特征提取结果进行控制
        latent_tensor = encode_tensor.view(b,c,th*tw)
        out,tensor1_afteratt = self.decoder(feature_dec,tensor4,tensor3,tensor2,tensor1)
        return out,latent_tensor,attn_weight,tensor1,tensor1_afteratt


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = Generator(in_ch=12,feature_root=16,out_ch=3).to(device)
    g_total, g_trainable = count_parameters(m)
    print(f"Generator parameters: {g_total:,} ({g_total / 1e6:.3f} M)")
    data = torch.rand(5, 12, 256, 256).to(device)
    out,_,attn,_,_ = m(data)
    print(out.shape)
