import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.nn import functional as F
from torch.autograd import Function

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, class_num = 940):
        super(ft_net, self).__init__()
        #load the model
        model_ft = models.densenet121(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        #num_ftrs = model_ft.fc.in_features
        modules = list(model_ft.children())[:-1]
        model_ft = nn.Sequential(*modules)
        # change avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.model = model_ft
        # self.classifier = ClassBlock(num_ftrs, class_num) #define our classifier.
        self.classifier = nn.Linear(num_ftrs, class_num)

    def forward(self, x):
        feat = self.model(x)
        feat = torch.squeeze(feat)
        output = self.classifier(feat) #use our classifier.
        return feat, output

class ft_net_test(nn.Module):
    def __init__(self):
        super(ft_net_test, self).__init__()
        #load the model
        model_ft = models.densenet121(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        #num_ftrs = model_ft.fc.in_features
        modules = list(model_ft.children())[:-1]
        model_ft = nn.Sequential(*modules)
        # change avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.model = model_ft
        # self.classifier = ClassBlock(num_ftrs, class_num) #define our classifier.

    def forward(self, x):
        feat = self.model(x)
        feat = torch.squeeze(feat)
        return feat

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
# Define the ResNet50-based Model
class DANN(nn.Module):
    def __init__(self, class_num = 940):
        super(DANN, self).__init__()
        #load the model
        model_ft = models.resnet34(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        modules = list(model_ft.children())[:-1]
        model_ft = nn.Sequential(*modules)
        # change avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.model = model_ft
        # self.classifier = ClassBlock(num_ftrs, class_num) #define our classifier.
        self.classifier = nn.Linear(num_ftrs, class_num)
        self.domain = nn.Linear(num_ftrs, 1)

    def forward(self, x, alpha):
        feat = self.model(x)
        feat = torch.squeeze(feat)
        feat_rev = ReverseLayerF.apply(feat, alpha)
        output = self.classifier(feat) #use our classifier.
        domain = self.domain(feat_rev)
        return feat, output, domain

def test():
    model = ft_net()
    img = torch.rand(3,3,384,192)
    output, _ = model(img)
    # for item in output:
    #     print(item.shape)
    print(output.size())

if __name__ == '__main__':
    test()
