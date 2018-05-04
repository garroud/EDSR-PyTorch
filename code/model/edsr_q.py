from model import common

import torch.nn as nn

def make_model(args, parent=False):
    return EDSR(args)

class EDSR_Q(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        window_size = args.window_size
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        #define process_q module
        m_process_q = [
            common.Process_q(n_feats, n_feats/2, n_feats, 3,kernel_size=kernel_size, window_si)
        ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.process_q = nn.Sequential(*m_process_q)
        self.corr= Correlation(pad_size=window_size, kernel_size=1, max_displacement=window_size, stride1=1, stride2=1, corr_multiply=1)
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        y = self.process_q(x)
        y = self.corr(y,y)
        y = self.build_q(y)
        x = self.add_mean(x)
        return x,y

    #construct the Q matrix based on correlation map
    def build_q(self,x):
        x_shape = x.shape
        batch = x_shape[0]
        channel = x_shape[1]
        height = x_shape[2]
        width = x_shape[3]
        build = torch.zeros(b,1,h*w,h*w)
        # shift = range(c) - window_size
        for b in range(batch):
            for c in range(channel):
                shift_x = c // (2*window_size+1) -window_size;
                shift_y = c % (2*window_size+1) - window_size;
                for h in range(height):
                    for w in range(width):
                        corr_x = x + shift_x
                        corr_y = y + shift_y
                        if corr_x >=0 and corr_y >=0 and corr_x < width and corr_y < height:
                            build[b,:,w+h*height,corr_x + corr_y * height] = x[b,c,h,w]
        return build

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
