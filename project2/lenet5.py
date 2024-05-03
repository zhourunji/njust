import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

BATCH_SIZE = 512  # 批量大小
EPOCHS = 5  # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为 GPU 或 CPU

# 下载训练集和测试集，并进行预处理
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),  # 调整图像大小为 32x32
                       transforms.ToTensor(),  # 将图像转换为张量，并归一化到[0,1]区间
                       transforms.Normalize((0.1037,), (0.3081,))  # 标准化图像，使均值为0.1037，标准差为0.3081
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)  # 设置批量大小和打乱数据顺序

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),  # 调整图像大小为 32x32
        transforms.ToTensor(),  # 将图像转换为张量，并归一化到[0,1]区间
        transforms.Normalize((0.1037,), (0.3081,))  # 标准化图像，使均值为0.1037，标准差为0.3081
    ])),
    batch_size=BATCH_SIZE, shuffle=True)  # 设置批量大小和打乱数据顺序

# 定义 LeNet-5 模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 第一卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)  # 第二卷积层
        self.fc1 = nn.Linear(5 * 5 * 16, 120)  # 第一全连接层
        self.fc2 = nn.Linear(120, 84)  # 第二全连接层
        self.fc3 = nn.Linear(84, 10)  # 输出层

    def forward(self, x):
        in_size = x.size(0)  # 输入数据的批量大小
        out = self.conv1(x)  # 第一卷积层
        out = F.relu(out)  # ReLU 激活函数
        out = F.max_pool2d(out, 2, 2)  # 最大池化层
        out = self.conv2(out)  # 第二卷积层
        out = F.relu(out)  # ReLU 激活函数
        out = F.max_pool2d(out, 2, 2)  # 最大池化层
        out = out.view(in_size, -1)  # 展平操作
        out = self.fc1(out)  # 第一全连接层
        out = F.relu(out)  # ReLU 激活函数
        out = self.fc2(out)  # 第二全连接层
        out = F.relu(out)  # ReLU 激活函数
        out = self.fc3(out)  # 输出层
        return out

# 创建 LeNet 模型和优化器
model = LeNet().to(DEVICE)  # 将模型移动到对应的设备上
optimizer = optim.Adam(model.parameters())  # Adam 优化器

# 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 梯度清零
        output = model(data)  # 前向传播
        loss = F.cross_entropy(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

# 定义测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum')  # 累计测试集损失
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的类别
            correct += pred.eq(target.view_as(pred)).sum().item()  # 计算正确分类数量

    test_loss /= len(test_loader.dataset)  # 平均测试集损失
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))

# 训练和测试模型
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)
