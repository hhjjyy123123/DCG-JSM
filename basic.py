import torch.nn as nn
import torch
import torch.nn.functional as F

class UpSample(nn.Module):
    '''
    Upsample the feature_map
    '''
    def __init__(self,in_ch,out_ch):
        super(UpSample,self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1)
    def forward(self,x):
        x = self.upsample(x)
        return x


class MergeModule(nn.Module):
    def __init__(self):
        super(MergeModule,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )

    def forward(self,x):
        avg_pool = torch.mean(x,dim=1,keepdim=True)
        max_pool,_ = torch.max(x,dim=1,keepdim=True)
        fea_concat = torch.concat([avg_pool,max_pool],dim=1)
        out = self.conv(fea_concat)
        return out



class ResSE(nn.Module):
    def __init__(self,in_ch,out_ch,reduction_rate):
        super(ResSE,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.hidden = in_ch // reduction_rate
        self.FC = nn.Sequential(
            nn.Linear(in_features=in_ch,out_features=self.hidden,bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.hidden,out_features=out_ch,bias=False),
            nn.Sigmoid()
        )

    def forward(self,input_tensor):
        """
        :param input_tensor:[b,3c,h,w]
        :return:[b,out_ch,1,1]
        """
        b,c,h,w = input_tensor.shape
        x_e = self.avg_pool(input_tensor).view(b,c)
        out = self.FC(x_e).view(b,-1,1,1)
        return out



class LocalizationAttention(nn.Module):
    def __init__(self,in_ch,out_ch,reduction=16):
        super(LocalizationAttention,self).__init__()

        self.avg_pool_h = nn.AdaptiveAvgPool2d((None,1))
        self.max_pool_h = nn.AdaptiveMaxPool2d((None,1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1,None))
        self.max_pool_w = nn.AdaptiveMaxPool2d((1,None))

        hidden_ch = max(8,in_ch // reduction)

        self.cv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=hidden_ch,kernel_size=1,stride=1,padding=0),
            nn.GroupNorm(num_groups=4,num_channels=hidden_ch),
            nn.SiLU(inplace=False)
        )

        self.conv_h = nn.Conv2d(in_channels=hidden_ch,out_channels=out_ch,kernel_size=1,stride=1,padding=0)
        self.conv_w = nn.Conv2d(in_channels=hidden_ch,out_channels=out_ch,kernel_size=1,stride=1,padding=0)

    def forward(self,feature):
        """
        :param feature:[b,3c,h,w]
        :return: feature_out:[b,c,h,w]
        """
        b,_,h,w = feature.size()
        feature_h = self.avg_pool_h(feature) + self.max_pool_h(feature)  #[b,3c,h,1]
        feature_w = self.avg_pool_w(feature) + self.max_pool_w(feature)  #[b,3c,1,w]
        feature_w = feature_w.permute(0,1,3,2)

        f_cat = torch.concat([feature_h,feature_w],dim=2)  #[b,3c,h+w,1]
        f_cat = self.cv(f_cat)  #[b,hidden_ch,h+w,1]

        feature_h_split,feature_w_split = torch.split(f_cat,[h,w],dim=2)
        feature_w_split = feature_w_split.permute(0,1,3,2)

        a_h = self.conv_h(feature_h_split)
        a_w = self.conv_w(feature_w_split)
        return a_h,a_w


class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        x = self.downsample(input)
        return x
class ResBlock_V3(nn.Module):
    def __init__(self,in_ch,out_ch,num_group,reduction_rate):
        super(ResBlock_V3,self).__init__()
        self.downsample = Downsample()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=num_group, num_channels=out_ch),
            nn.ReLU(inplace=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_group, num_channels=out_ch),
            nn.ReLU(inplace=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(num_groups=num_group, num_channels=out_ch),
            nn.ReLU(inplace=False)
        )

        self.localAtt = LocalizationAttention(in_ch=out_ch*3,out_ch=out_ch,reduction=reduction_rate)

        self.skipconv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.GroupNorm(num_groups=num_group,num_channels=out_ch),
            nn.ReLU(inplace=False)
        )


    def forward(self,input_tensor):
        input_tensor_down = self.downsample(input_tensor)
        x1 = self.conv1(input_tensor_down)
        x2 = self.conv2(input_tensor_down)
        x3 = self.conv3(input_tensor_down)

        x_concat = torch.concat([x1,x2,x3],dim=1)  #[b,3*out_ch,h,w]
        att_h,att_w = self.localAtt(x_concat)   #[b,out_ch,h,1],[b,out_ch,1,w]
        x_fuse = (x1+x2+x3) * att_h * att_w

        x_skip = self.skipconv(input_tensor)
        out = F.relu(x_skip + x_fuse)
        return out

class ResNeXtBlockFixed(nn.Module):
    def __init__(self, channels, division=32,num_group=16):
        super(ResNeXtBlockFixed, self).__init__()
        mid_channels = max(channels // 2, division)

        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1,bias=False),
            nn.GroupNorm(num_group, mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1,
                      padding=1, groups=division, bias=False),
            nn.GroupNorm(num_group, mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_group, channels)
        )

        self.final_relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.final_relu(x + self.conv_block(x))



class TemporalChannelAttention(nn.Module):
    def __init__(self,in_ch,reduction=8):
        super(TemporalChannelAttention,self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=2*in_ch,out_features=in_ch//reduction,bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=in_ch//reduction,out_features=in_ch,bias=False)
        )

    def forward(self,F_cat):

        """
        :param F_cat:[b,c,h,w]
        :return:
        """
        B,C,_,_ = F_cat.shape
        mean_cat = F.adaptive_avg_pool2d(F_cat,1).view(B,C)
        var_cat = F.adaptive_avg_pool2d((F_cat-mean_cat.view(B,C,1,1))**2,1).view(B,C)
        std = torch.sqrt(var_cat+1e-7)
        attn = torch.sigmoid(self.mlp(torch.concat([mean_cat,std],dim=1)))
        return attn.view(B,C,1,1)

class TemporalSpatialAttention(nn.Module):
    def __init__(self,kernel_size=5):
        super(TemporalSpatialAttention,self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False)


    def forward(self,F_cat):
        """
        :param F_cat:[b,c,h,w]
        """
        mean_cat = torch.mean(F_cat,dim=1,keepdim=True) #[b,1,h,w]
        var_cat = torch.mean((F_cat-mean_cat)**2,dim=1,keepdim=True) #[b,1,h,w]
        std = torch.sqrt(var_cat+1e-7)
        attn = torch.sigmoid(self.conv(torch.concat([mean_cat,std],dim=1)))
        return attn

class TDCBAM(nn.Module):
    def __init__(self,in_ch,reduction=8,kernel_size=5):
        super(TDCBAM,self).__init__()
        self.ca = TemporalChannelAttention(in_ch=in_ch,reduction=reduction)
        self.sa = TemporalSpatialAttention(kernel_size=kernel_size)

    def forward(self,F_cat):

        A_c = self.ca(F_cat)   #[b,c,1,1]
        A_s = self.sa(F_cat)   #[b,1,h,w]
        A = A_c * A_s
        out = F_cat * A
        return out

class GatedCBAM_V2(nn.Module):
    def __init__(self,in_ch,reduction=8,kernel_size=5):
        super(GatedCBAM_V2,self).__init__()

        self.w_e = nn.Conv2d(in_channels=in_ch,out_channels=in_ch,kernel_size=1,stride=1,padding=0)
        self.w_d = nn.Conv2d(in_channels=in_ch,out_channels=in_ch,kernel_size=1,stride=1,padding=0)

        self.activate = nn.ReLU(inplace=False)
        self.ca = TemporalChannelAttention(in_ch=in_ch, reduction=reduction)
        self.sa = TemporalSpatialAttention(kernel_size=kernel_size)

        self.dist_conv = nn.Conv2d(in_channels=in_ch,out_channels=in_ch,kernel_size=1,stride=1,padding=0)

    def forward(self,f_encoder,f_decoder):

        f_encoder_conv = self.w_e(f_encoder)
        f_decoder_conv = self.w_d(f_decoder)

        f_dist = self.dist_conv(torch.abs(f_encoder-f_decoder))

        f_fused = self.activate(f_encoder_conv+f_decoder_conv+f_dist)

        A_c = self.ca(f_fused)
        A_s = self.sa(f_fused)
        A = A_c*A_s
        out = f_encoder * A
        return out


