import common
import numpy as np
import torch.nn as nn
import torch
import math
import sys
from correlation_package.modules.correlation import Correlation

def make_model(args, parent=False):
    return EDSR_Q(args)

class EDSR_Q(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR_Q, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.window_size = args.window_size
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
        ]
        m_result = [
        conv(n_feats, args.n_colors, kernel_size)
        ]
        #define process_q module
        m_process_q = [
            common.Process_q(n_feats, n_feats, kernel_size=3),
        ]
        m_process_v = [
            common.Process_q(n_feats, n_feats, kernel_size=3),
        ]
        m_process_v.append(conv(n_feats,1,kernel_size))

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.build_q = build_q
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.result = nn.Sequential(*m_result)
        self.process_q = nn.Sequential(*m_process_q)
        # self.process_v = nn.Sequential(*m_process_v)
        self.corr= Correlation(pad_size=self.window_size, kernel_size=1, max_displacement=self.window_size, stride1=1, stride2=1, corr_multiply=1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        y = self.process_q(x)
        # v = torch.abs(self.process_v(x))
        # z0 = self.corr(self.cal_mean(y),self.cal_mean(y))
        # y = torch.mul(self.corr(y,y), torch.rsqrt(z0))
        y = self.corr(y,y)
        # print 'begin'
        y = self.build_q(self.window_size,y)
        # print 'done'
        # y = self.combine_q(y,v)
        x = self.result(x)
        x = self.add_mean(x)
        return x,y

    def combine_q(self, q0, v):
        b = v.shape[0]
        v = v.squeeze(1).view(b,-1,1)
        v = torch.bmm(v, torch.transpose(v,1,2))
        print (q0[0,0,10:15,10:15])
        print (v[0,10:15,10:15])
        test = q0 * v.unsqueeze(1)
        print test[0,0,10:15,10:15]
        # TODO: some wrong here
        return test

    def cal_mean(self, x):
        x = torch.mul(x,x)
        return torch.mean(x, 1, True)
    #construct the Q matrix based on correlation map
    # def build_q(self,window_size,x):
    #     print x[0,2,10:15,10:15]
    #     print x.shape
    #     # window_size = 1
    #     # print window_size
    #     x_shape = x.shape
    #     batch = x_shape[0]
    #     channel = x_shape[1]
    #     height = x_shape[2]
    #     width = x_shape[3]
    #     build = torch.zeros(batch,1,height*width,height*width,requires_grad=True).cuda()
    #     # shift = range(c) - window_size
    #     for b in range(batch):
    #         for c in range(channel):
    #             shift_x = c % (2*window_size+1) -window_size;
    #             shift_y = c // (2*window_size+1) - window_size;
    #             for h in range(height):
    #                 for w in range(width):
    #                     corr_x = w + shift_x
    #                     corr_y = h + shift_y
    #                     if corr_x >=0 and corr_y >=0 and corr_x < width and corr_y < height:
    #                         index = (torch.tensor(b),torch.tensor(0),torch.tensor(h*width+w),torch.tensor(corr_y*width+corr_x))
    #                         build.index_put_(index,x[b,c,h,w])
    #                         # print (build[b,0,h*width+w:h*width+5,corr_y*width+corr_x:corr_y*width+corr_x+5])
    #         test = build[b,0,:,:]
    #         print test.shape
    #         print test[10:15,10:15]
    #         if not torch.equal(build[b,0,:,:],build[b,0,:,:].t()):
    #             print 'shit error'
    #         # e,v = torch.symeig(test)
    #         # print torch.min(e)
    #         #not sure why cholesky decomposition failed
    #     # torch.btrifact(build[:,0,:,:],False)
    #     # print P.size
    #     # print A_L[0,10:15,10:15]

        # print torch.min(build.view(-1,1))
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

def build_q(window_size, x):
    # print x[0,2,10:15,10:15]
    # window_size = 1
    x_shape = x.shape
    batch = x_shape[0]
    channel = x_shape[1]
    height = x_shape[2]
    width = x_shape[3]
    build = torch.zeros(batch,1,height*width,height*width,requires_grad=True).cuda()
    # shift = range(c) - window_size
    for b in range(batch):
        for c in range(channel):
            shift_x = c % (2*window_size+1) -window_size;
            shift_y = c // (2*window_size+1) - window_size;
            for h in range(height):
                for w in range(width):
                    corr_x = w + shift_x
                    corr_y = h + shift_y
                    if corr_x >=0 and corr_y >=0 and corr_x < width and corr_y < height:
                        index = (torch.tensor(b),torch.tensor(0),torch.tensor(h*width+w),torch.tensor(corr_y*width+corr_x))
                        # print c
                        # print index
                        build.index_put_(index,x[b,c,h,w])
                        # print (build[b,0,h*width+w:h*width+5,corr_y*width+corr_x:corr_y*width+corr_x+5])
        # if not torch.equal(build[b,0,:,:],build[b,0,:,:].t()):
        #     print 'shit error'
        # test = torch.mm(build[b,0,:,:], build[b,0,:,:].t())
        # u = torch.potrf(test)
        # print b
        # print u
        # e,v = torch.symeig(test)
        # print torch.min(e)
        #not sure why cholesky decomposition failed
    # torch.btrifact(build[:,0,:,:],False)
    # print P.size
    # print A_L[0,10:15,10:15]

    # print torch.min(build.view(-1,1))
    return build

if __name__ == '__main__':
    # test_x = torch.tensor([x+1 for x in range(9)],dtype=torch.float32)
    # test_x = test_x.view(1,1,3,3)
    # test_x = torch.cat((test_x, test_x+1), 1)
    test_x = torch.rand(1,16,48,48)
    corr= Correlation(pad_size=3, kernel_size=1, max_displacement=3, stride1=1, stride2=1, corr_multiply=1).cuda()
    test_xs = corr(test_x.cuda(), test_x.cuda())
    # for i in range(8)
    print test_x.shape
    print test_xs.shape
    # print test_xs
    # print 'let\'s build the q'
    # Not work after testing
    result = torch.squeeze(build_q(3,test_xs))

    print result.shape
    # print result
    # print torch.equal(result, result.t())
    print torch.equal(result, result.t())
    test = torch.mm(result,result)
    e,v = torch.symeig(test)
    print torch.min(e)
    u = torch.potrf(test)
    print u
