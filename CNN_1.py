#1.导入相关库
import torch
import torch.nn as nn
from torchvision import models
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
import scipy.io as scio

#2.读取数据集
train_data_path = r'D:\data\training\training'
val_data_path = r'D:\data\validation\validation'


#3.预处理
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = ImageFolder(train_data_path, transform=data_transform)

train_data_loader = Data.DataLoader(
    train_data,
    batch_size=2,
    shuffle=True,
    # num_workers=2,
)

val_data = ImageFolder(val_data_path, transform=data_transform)

val_data_loader = Data.DataLoader(
    val_data,
    batch_size=2,
    shuffle=True,
    # num_workers=2,
)


#4.构建卷积神经网络模型
# Alexnet ,VGG, RESNET

vgg16 =models.vgg16(pretrained=True)
vgg_feature = vgg16.features

class MYVGG16MODULE(nn.Module):
    def __init__(self):
        super(MYVGG16MODULE, self).__init__()
        self.vgg = vgg_feature
        self.classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.Tanh(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Dropout(p=0.3),

            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.vgg(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


my_vgg = MYVGG16MODULE()
#5。训练网络

optimizer = torch.optim.SGD(my_vgg.parameters(), lr=0.003)
loss_fun = nn.CrossEntropyLoss()

epoches = 1

train_loss_epoch = []
train_corretc_epoch = []


val_loss_epoch = []
val_correct_epoch = []

for epoch in range(epoches):
    train_loss = 0
    train_correct = 0
    for step, (b_x, b_y) in enumerate(train_data_loader):
        output = my_vgg(b_x)
        loss = loss_fun(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pre_lable = torch.argmax(output, 1)
        train_correct += torch.sum(pre_lable == b_y)

    train_loss_epoch.append(train_loss_epoch)
    train_corretc_epoch.append(train_correct / len(train_data))

    print('训练完成', epoch+1, '轮次')


    my_vgg.eval()
    val_loss = 0
    val_correct = 0
    for step, (val_x, val_y) in enumerate(val_data_loader):
        output = my_vgg(val_x)
        loss = loss_fun(output, val_y)

        pre_lable = torch.argmax(output, 1)
        val_loss += loss.item()
        val_correct += torch.sum(pre_lable==b_y)

    val_loss_epoch.append(val_loss)
    val_correct_epoch.append(val_correct / len(val_data))

#6.模型保存

torch.save(my_vgg, 'myvgg1.pkl', _use_new_zipfile_serialization=False)
scio.savemat('val_loss.mat', {'val_loss':val_loss})


