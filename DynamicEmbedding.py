import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):
    def __init__(self,in_ch,hidden_ch,num_group,kernel_size=3):
        super(ConvGRUCell,self).__init__()
        self.hidden_ch = hidden_ch
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels=in_ch+hidden_ch,out_channels=2*hidden_ch,kernel_size=kernel_size,padding=padding)
        self.gn1 = nn.GroupNorm(num_groups=num_group,num_channels=2*hidden_ch)

        self.conv2 = nn.Conv2d(in_channels=in_ch+hidden_ch,out_channels=hidden_ch,kernel_size=kernel_size,padding=padding)
        self.gn2 = nn.GroupNorm(num_groups=num_group,num_channels=hidden_ch)


    def forward(self,x,h_perv):
        if h_perv is None:
            b,_,h,w = x.shape
            h_perv = torch.zeros(b,self.hidden_ch,h,w).to(x.device)

        combined = torch.concat([x,h_perv],dim=1)
        conv_gates = self.gn1(self.conv1(combined))
        z,r = torch.split(conv_gates,self.hidden_ch,dim=1)
        z,r = torch.sigmoid(z),torch.sigmoid(r)

        combined_2 = torch.concat([x,r*h_perv],dim=1)
        h_t = torch.tanh(self.gn2(self.conv2(combined_2)))

        h_next = (1-z)*h_perv+z*h_t

        return h_next


class ConvGRU(nn.Module):
    def __init__(self,in_ch,hidden_ch,num_group,num_frame=4,num_layers=4):
        super(ConvGRU,self).__init__()
        self.num_frame = num_frame
        self.hidden_ch = hidden_ch
        self.num_layers = num_layers
        self.input_per_frame_ch = in_ch // num_frame

        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            cur_in_ch = self.input_per_frame_ch if i==0 else hidden_ch
            self.cell_list.append(ConvGRUCell(in_ch=cur_in_ch,hidden_ch=hidden_ch,num_group=num_group))

        self.reverse = nn.Conv2d(in_channels=hidden_ch,out_channels=in_ch,kernel_size=1,stride=1)

    def forward(self,input_x):
        b,c,h,w = input_x.shape

        x_seq = input_x.view(b,self.num_frame,self.input_per_frame_ch,h,w)
        hidden_states = [None] * self.num_layers

        for t in range(self.num_frame):
            x_t = x_seq[:,t,:,:,:]
            cur_input = x_t
            for layer in range(self.num_layers):
                h_prev = hidden_states[layer]
                h_cur = self.cell_list[layer](cur_input,h_prev)
                hidden_states[layer] = h_cur
                cur_input = h_cur

        final_out = self.reverse(hidden_states[-1])
        return final_out

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = ConvGRU(in_ch=256,hidden_ch=64,num_group=8,num_frame=4,num_layers=4).to(device)
    data = torch.rand(5, 256, 16, 16).to(device)
    out = m(data)
    print(out.shape)



