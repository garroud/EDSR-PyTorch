import common
import numpy as np
import torch.nn as nn
import torch
import math
import sys
import time
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
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
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
        self.corr_index, self.corr_x, self.corr_y = self.get_index()
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
        # y = self.build_q(self.window_size,y)
        y = self.build_q_sparse(y,self.corr_index, self.corr_x, self.corr_y)
        # print 'done'
        # y = self.combine_q(y,v)
        x = self.result(x)
        x = self.add_mean(x)
        return x,y

    def get_index(self):
        window_size = self.window_size
        batch = self.batch_size
        channel = (self.window_size *2 + 1) **2
        height = self.patch_size
        width = self.patch_size
        # x = x.view(batch, -1)
        index = torch.tensor(range(height*width),dtype=torch.int64, requires_grad=False)
        index = index.repeat(1,channel)
        h_index = index // width
        w_index = index % width
        channels = torch.tensor(range(channel),dtype=torch.int64, requires_grad=False)
        channels = channels.repeat(width*height,1).t().contiguous().view(1,-1)
        shift_x = channels % (2*window_size+1) - window_size
        shift_y = channels //(2*window_size+1) - window_size
        corr_x = w_index + shift_x
        corr_y = h_index + shift_y
        corr_index = torch.tensor([[index[0,i],corr_y[0,i]*height+corr_x[0,i]] for i in range(index.shape[1]) \
                       if(corr_x[0,i]>=0 and corr_x[0,i]< width and corr_y[0,i]>= 0 and corr_y[0,i] < height)],dtype=torch.int64, requires_grad=False)
        corr_index = corr_index.t()
        #make batch index
        batch_index = torch.tensor(range(batch),dtype=torch.int64,requires_grad=False).repeat(corr_index.shape[1],1).t().contiguous().view(1,-1)
        corr_index = torch.cat((batch_index,corr_index.repeat(1,batch)),dim=0)
        print corr_index.shape
        return corr_index, corr_x, corr_y

    def build_q_sparse(self, x, corr_index, corr_x, corr_y):
        batch = x.shape[0]
        channel = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        x_t = x.view(batch, -1).t()
        data_avaliable = torch.tensor([ x_t[i,:].tolist() for i in range(corr_x.shape[1]) \
            if(corr_x[0,i] >=0 and corr_x[0,i]< width and corr_y[0,i]>= 0 and corr_y[0,i] < height)], requires_grad=True)
        # print data_avaliable.shape
        build = torch.sparse.FloatTensor(corr_index, data_avaliable.t().contiguous().view(1,-1).squeeze(),torch.Size([batch, height*width, height*width])).to_dense().cuda()
        return torch.unsqueeze(build,1)

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
#old version to build matrix, test case not used in the real implementation
def build_q(window_size, x):
    x_shape = x.shape
    batch = x_shape[0]
    channel = x_shape[1]
    height = x_shape[2]
    width = x_shape[3]
    build = torch.zeros(batch,1,height*width,height*width,requires_grad=True).cuda()
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
                        build.index_put_(index,x[b,c,h,w])
    return build


def build_q_sparse(window_size, x):
    x_shape = x.shape
    batch = x_shape[0]
    channel = x_shape[1]
    height = x_shape[2]
    width = x_shape[3]
    x = x.view(batch, -1)
    index = torch.tensor(range(height*width),dtype=torch.int64, requires_grad=False)
    index = index.repeat(1,channel)
    h_index = index // width
    w_index = index % width
    channels = torch.tensor(range(channel),dtype=torch.int64, requires_grad=False)
    channels = channels.repeat(width*height,1).t().contiguous().view(1,-1)
    shift_x = channels % (2*window_size+1) - window_size
    shift_y = channels //(2*window_size+1) - window_size
    corr_x = w_index + shift_x
    corr_y = h_index + shift_y
    corr_index = torch.tensor([[index[0,i],corr_y[0,i]*height+corr_x[0,i]] for i in range(index.shape[1]) \
                   if(corr_x[0,i]>=0 and corr_x[0,i]< width and corr_y[0,i]>= 0 and corr_y[0,i] < height)],dtype=torch.int64, requires_grad=False)
    corr_index = corr_index.t()
    #make batch index
    batch_index = torch.tensor(range(batch),dtype=torch.int64,requires_grad=False).repeat(corr_index.shape[1],1).t().contiguous().view(1,-1)
    corr_index = torch.cat((batch_index,corr_index.repeat(1,batch)),dim=0)
    print corr_index.shape
    x_t = x.t()
    data_avaliable = torch.tensor([ x_t[i,:].tolist() for i in range(index.shape[1]) \
        if(corr_x[0,i] >=0 and corr_x[0,i]< width and corr_y[0,i]>= 0 and corr_y[0,i] < height)])
    print data_avaliable.shape
    build = torch.sparse.FloatTensor(corr_index, data_avaliable.t().contiguous().view(1,-1).squeeze(),torch.Size([batch, height*width, height*width])).to_dense()
    return build

if __name__ == '__main__':
    # test_x = torch.tensor([x+1 for x in range(9)],dtype=torch.float32)
    # test_x = test_x.view(1,1,3,3)
    # test_x = torch.cat((test_x, test_x+1), 1)
    test_x = torch.rand(2,16,48,48)
    corr= Correlation(pad_size=3, kernel_size=1, max_displacement=3, stride1=1, stride2=1, corr_multiply=1).cuda()
    test_xs = corr(test_x.cuda(), test_x.cuda())
    # for i in range(8)
    print test_x.shape
    print test_xs.shape
    # print test_xs
    # print 'let\'s build the q'
    # Not work after testing
    a = time.time()
    result = torch.squeeze(build_q(3,test_xs))
    b = time.time()
    print b-a
    result_sparse = build_q_sparse(3,test_xs)
    c = time.time()
    print c-b
    print result.shape
    print result_sparse.shape
    print torch.equal(result, result_sparse)
    # print result
    # print torch.equal(result, result.t())
    # print torch.equal(result, result.t())
    # test = torch.mm(result,result)
    # e,v = torch.symeig(test)
    # print torch.min(e)
    # u = torch.potrf(test)
