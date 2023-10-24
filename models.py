import torch.nn as nn




def conv_block(in_channels, out_channels):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    
    return nn.Sequential(*layers)




class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64) # 64 X 32 X 32
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        self.res2 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        self.net_work_1 = conv_block(64, 128)
        self.max_pool1 = nn.MaxPool2d(2) # 128 X 16 X 16
        self.res3 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.res4 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.net_work_2 = conv_block(128, 256)
        self.max_pool2 = nn.MaxPool2d(2) # 256 X 8 X 8
        self.res5 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.res6 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.net_work_3 = conv_block(256, 512)
        self.max_pool3 = nn.MaxPool2d(2) # 512 X 4 X 4
        self.res7 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.res8 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(2), # 512 x 2 x 2
                                        nn.Flatten(), # 2048
                                        # nn.Dropout(0.5),
                                        nn.Linear(2048, num_classes)) # 2048 -> 100
        
        
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.res2(out) + out 
        out = self.max_pool1(out)
        out = self.net_work_1(out)
        
        out = self.res3(out) + out
        out = self.res4(out) + out 
        out = self.max_pool2(out)
        out = self.net_work_2(out)
        
        out = self.res5(out) + out
        out = self.res6(out) + out 
        out = self.max_pool3(out)
        out = self.net_work_3(out)
        
        out = self.res7(out) + out
        out = self.res8(out) + out 
        out = self.classifier(out)
        
        return out