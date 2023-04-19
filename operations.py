import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        #print(y)
        return x * y.expand_as(x)


OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: AvgPoolLayer(C, stride, affine),
  'max_pool_3x3' : lambda C, stride, affine: MaxPoolLayer(C, stride, affine),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  #'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  #'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
  #  nn.ReLU(inplace=False),
  #  nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
  #  nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
  #  nn.BatchNorm2d(C, affine=affine)
  #  ),
}

class AvgPoolLayer(nn.Module):
    def __init__(self, C, stride, affine, splits=4):
        super(AvgPoolLayer, self).__init__()
        #self.ith = -1
        self.splits = splits
        self.avgpool = nn.ModuleList([nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False).cuda() for _ in range(splits)])

    def forward(self, x):
        dim_2 = x.shape[1]
        ans = []
        for i in range(self.splits):
            ans.append(self.avgpool[i](x[:, i*dim_2//self.splits: (i+1)*dim_2//self.splits]))

        #self.ith = (self.ith+1)%self.splits
        #print(self.ith)
        #return self.avgpool[self.ith](x)
        return torch.cat(ans, 1)

class MaxPoolLayer(nn.Module):
    def __init__(self, C, stride, affine, splits=4):
        super(MaxPoolLayer, self).__init__()
        self.splits = splits
        self.maxpool = nn.ModuleList([nn.MaxPool2d(3, stride=stride, padding=1).cuda() for _ in range(splits)])

    def forward(self, x):
        dim_2 = x.shape[1]
        ans = []
        for i in range(self.splits):
            ans.append(self.maxpool[i](x[:, i*dim_2//self.splits: (i+1)*dim_2//self.splits]))
        #self.ith = (self.ith+1)%self.splits
        #print(self.ith)

        #return self.maxpool[self.ith](x)
        return torch.cat(ans, 1)

class ReLUConvBNOrigin(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBNOrigin, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            #nn.PReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class ReLUConvBN(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, splits=4, affine=True):
    super(ReLUConvBN, self).__init__()
    self.splits = splits
    #self.op = nn.Sequential(
    self.relu = nn.ModuleList([nn.ReLU(inplace=False).cuda() for _ in range(splits)])
      #nn.PReLU(),
    self.conv = nn.ModuleList([nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False).cuda() for _ in range(splits)])
    self.bn = nn.ModuleList([nn.BatchNorm2d(C_out, affine=affine).cuda() for _ in range(splits)])
    #).cuda()

  def forward(self, x):
    dim_2 = x.shape[1]
    ans = []
    for i in range(self.splits):
        out = self.relu[i](x[:, i*dim_2//self.splits: (i+1)*dim_2//self.splits])
        out = self.conv[i](out)
        out = self.bn[i](out)
    return torch.cat(ans, 1)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, splits=4, affine=True):
    super(DilConv, self).__init__()
    self.splits = splits
    #self.op = nn.Sequential(
    self.relu = nn.ModuleList([nn.ReLU(inplace=False).cuda() for _ in range(splits)])
    #nn.PReLU(),
    self.conv_1 = nn.ModuleList([nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False).cuda() for _ in range(splits)])
    self.conv_2 = nn.ModuleList([nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False).cuda() for _ in range(splits)])
    self.bn = nn.ModuleList([nn.BatchNorm2d(C_out, affine=affine).cuda() for _ in range(splits)])
    #)
    self.se = nn.ModuleList([SELayer(C_out) for _ in range(splits)])
  def forward(self, x):
    dim_2 = x.shape[1]
    ans = []
    for i in range(self.splits):
        out = self.relu[i](x[:, i*dim_2//self.splits: (i+1)*dim_2//self.splits])
        out = self.conv_1[i](out)
        out = self.conv_2[i](out)
        out = self.bn[i](out)
        out = self.se[i](out)
        ans.append(out)
    #out = self.se[ith](out)
    #self.ith = (self.ith+1)%self.splits
    #print(self.ith)
    return torch.cat(ans, 1)
    #return out


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, splits=4, affine=True):
    super(SepConv, self).__init__()
    self.splits= splits
    #self.op = nn.Sequential(
    self.relu = nn.ModuleList([nn.ReLU(inplace=False) for _ in range(splits)])
    #nn.PReLU(),
    self.conv_1 = nn.ModuleList([nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False).cuda() for _ in range(splits)])
    self.conv_2 = nn.ModuleList([nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False).cuda() for _ in range(splits)])
    self.bn_1 = nn.ModuleList([nn.BatchNorm2d(C_in, affine=affine).cuda() for _ in range(splits)])
    self.relu2 = nn.ModuleList([nn.ReLU(inplace=False).cuda() for _ in range(splits)])
    #nn.PReLU(),
    self.conv_3 = nn.ModuleList([nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False).cuda() for _ in range(splits)])
    self.conv_4 = nn.ModuleList([nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False).cuda() for _ in range(splits)])
    self.bn_2 = nn.ModuleList([nn.BatchNorm2d(C_out, affine=affine).cuda() for _ in range(splits)])
    self.se = nn.ModuleList([SELayer(C_out) for _ in range(splits)])

  def forward(self, x):
    dim_2 = x.shape[1]
    ans = []
    for i in range(self.splits):
        out = self.relu[i](x[:, i*dim_2//self.splits: (i+1)*dim_2//self.splits])
        out = self.conv_1[i](out)
        out = self.conv_2[i](out)
        out = self.bn_1[i](out)
        out = self.relu2[i](out)
        out = self.conv_3[i](out)
        out = self.conv_4[i](out)
        out = self.bn_2[i](out)
        out = self.se[i](out)
        ans.append(out)
    #return out
    return torch.cat(ans, 1)


class Identity(nn.Module):

  def __init__(self, splits=4):
    super(Identity, self).__init__()
    self.splits= splits

  def forward(self, x):
    #self.ith = (self.ith+1)%self.splits

    #print(self.ith)

    return x


class Zero(nn.Module):

  def __init__(self, stride, splits=4):
    super(Zero, self).__init__()
    self.splits =splits
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    #print(self.ith)

    return x[:,:,::self.stride,::self.stride].mul(0.)

class FactorizedReduceOrigin(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduceOrigin, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

                                
    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, splits=4, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.splits = splits
    self.relu = nn.ModuleList([nn.ReLU(inplace=False).cuda() for _ in range(splits)])
    self.conv_1 = nn.ModuleList([nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False).cuda() for _ in range(splits)])
    self.conv_2 = nn.ModuleList([nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False).cuda() for _ in range(splits)])
    self.bn = nn.ModuleList([nn.BatchNorm2d(C_out, affine=affine) for _ in range(splits)])

  def forward(self, x):
    dim_2 = x.shape[1]
    ans = []
    for i in range(self.splits):
        out = self.relu[i](x[:, i*dim_2//self.splits: (i+1)*dim_2//self.splits])
        out = torch.cat([self.conv_1[i](out), self.conv_2[i](out[:,:,1:,1:])], dim=1)
        out = self.bn[i](out)
        ans.append(out)
    #self.ith = (self.ith+1)%self.splits
    #print(self.ith)
    #return out
    return torch.cat(ans, 1)
