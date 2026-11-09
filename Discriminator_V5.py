import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(self, img_in_ch=3, feature_root=32):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(img_in_ch, feature_root, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature_root, feature_root*2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature_root*2, feature_root*4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature_root*4, feature_root*8,  kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.classifier = spectral_norm(nn.Conv2d(feature_root*8, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, pred_img, return_feature=False):
        """
        :param pred_img: [b, 3, 256, 256]
        :return out: [b, 1, 30, 30]
        """
        x1 = self.conv1(pred_img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        out = self.classifier(x4)

        if return_feature:
            return out, x4
        else:
            return out

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = Discriminator().to(device)
    d_total, d_trainable = count_parameters(m)
    print(f"dis parameters: {d_total:,} ({d_total / 1e6:.3f} M)")
    data1 = torch.rand(5,3,256,256).to(device)
    out,fea = m(data1,True)
    print(out.shape)