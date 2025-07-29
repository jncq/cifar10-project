import torchvision
import torch
import torch.utils.data
import torchvision.transforms as tf
import torch.optim as op


"""加载并正则化CIFAR10数据集"""

# 封装一组转换函数对象作为转换器
# 在transform中添加数据增强和分析
transform_train = tf.Compose([
    tf.RandomHorizontalFlip(),  # 随机水平翻转（增强数据多样性）
    tf.RandomRotation(30),      # 随机旋转±30度
    tf.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 添加颜色抖动
    tf.ToTensor(),              # ToTensor()类把PIL Image格式的图片和Numpy数组转换成张量
    tf.RandomErasing(p=0.5, scale=(0.02, 0.1), value='random'),  # 添加随机遮挡增强泛化能力（Cutout变种）
    tf.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),   # 用均值和标准差归一化张量图像
    ])

# 验证集使用基础预处理
transform_val = tf.Compose([
    tf.ToTensor(),
    tf.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)) 
    ])

# 加载完整训练集
full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=None)

# 划分训练集和验证集（90%训练，10%验证）
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(full_dataset, 
    [train_size, val_size],generator=torch.Generator().manual_seed(8102240106))

# 应用不同的transform
class ApplyTransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)

train_set = ApplyTransformSubset(train_subset, transform_train)
val_set = ApplyTransformSubset(val_subset, transform_val)

# 可视化用的小批量加载器
vis_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=3)
# 训练用的大批量加载器
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True,num_workers=3,
                                           pin_memory=True,persistent_workers=True,prefetch_factor=1)
# 创建验证集加载器
val_loader = torch.utils.data.DataLoader(val_set,batch_size=128,shuffle=False,num_workers=2,
                                         pin_memory=True,persistent_workers=True)

# CIFAR10数据集所有类别名称
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""
训练可视化
"""

import matplotlib.pyplot as plt
import numpy as np

def img_show(img,n=4):
    img = img * 0.25 + 0.5  # 反正则化
    npimg = img.numpy()     # 转换成Numpy数组
    plt.figure(figsize=(8, 3))  # 设置合适画布尺寸
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')  # 关闭坐标轴
    
    # 生成标签文本并添加到图像下方
    label_text = "    ".join(f'{classes[vis_labels[j]]:5s}' for j in range(n))
    plt.figtext(0.5, 0.05, label_text, ha='center', va='bottom', fontsize=12)  # 在画布底部中心添加文本
    plt.show()
"""
定义神经网络
"""
import torch.nn as nn
import torch.nn.functional as F

# 定义网络模型，继承自torch.nn.Module类
class Model(nn.Module):
    def __init__(self):
        super().__init__()  # 初始化父类的属性
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 卷积层1
        self.bn1 = nn.BatchNorm2d(32)  # 批量归一化加速收敛
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 卷积层2
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  #卷积层3
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)  #卷积层4
        self.bn4 = nn.BatchNorm2d(256)
        
        # 添加全局平均池化代替全连接
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 10)  # 减少全连接层复杂度

        # 使用LeakyReLU代替ReLU
        self.activate = nn.LeakyReLU(0.1)


    def forward(self, x):
        x = self.activate(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = self.activate(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = self.activate(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = self.activate(self.bn4(self.conv4(x)))
        x = self.gap(x)
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

"""
训练模型
"""
#开始训练
if __name__ == '__main__':
    train_losses = []  # 存储每个epoch的训练损失
    val_losses = []    # 存储每个epoch的验证损失
    val_accuracies = []  # 存储每个epoch的验证准确率

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #自动检测系统是否支持CUDA，优先使用GPU
    print(f"可用设备: {torch.cuda.device_count()} GPU(s)")
    if device.type == 'cuda':
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")

    # 从 vis_loader 加载小批量（4个样本）用于显示
    vis_iter = iter(vis_loader)
    vis_images, vis_labels = next(vis_iter)  # 直接获取CPU上的预处理数据

    # 显示图像和标签
    img_show(torchvision.utils.make_grid(vis_images))
    

    # 实例化模型对象
    model = Model()
    model.to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器定义（AdamW）
    opizer = op.AdamW(model.parameters(), lr=0.001, weight_decay=0.005)
    # 在优化器定义后添加学习率调度器，余弦退火提升收敛稳定性
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opizer, T_max=100)  

    print("***********开始训练*************")

    # 在训练开始前初始化早停相关变量
    best_val_loss = float('inf')  # 初始最佳验证损失设为无穷大
    epochs_no_improve = 0  # 连续未改善的epoch计数器

    for epoch in range(10):  # 100次循环
        model.train()

        total_train_loss = 0.0
        total_samples = 0

        running_loss = 0.0  # 损失函数记录    
        for i, data in enumerate(train_loader, 0):
            # 获取模型输入；data是由[inputs, labels]组成的列表
            inputs, labels = data #把参数的梯度清零
            #数据迁移到设备
            inputs = inputs.to(device, non_blocking=True)  #异步传输
            labels = labels.to(device, non_blocking=True)  #将数据从CPU内存复制到GPU显存
            opizer.zero_grad()
            # 前向传播+反向传播+更新权重
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opizer.step()

            batch_size = inputs.size(0)
            total_train_loss += loss.item() * batch_size
            total_samples += batch_size

            # 打印统计数据
            running_loss += loss.item()
            if i % 100 == 99:  # 每100个batch打印一次统计信息
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.5f}')
                running_loss = 0.0  # 把损失函数记录清零
        
        epoch_train_loss = total_train_loss / total_samples
        train_losses.append(epoch_train_loss)
        print(f'Epoch {epoch+1}，训练损失: {epoch_train_loss:.5f}')

        model.eval()  # 设置为评估模式
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():  # 禁用梯度计算
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)  # 前向传播
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)  # 累加批次损失
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_set)  # 计算平均验证损失
        val_acc = 100 * correct_val / total_val

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 打印验证信息
        print(f'Epoch {epoch+1}，验证集损失: {val_loss:.5f}，验证准确率: {val_acc:.2f}%')
        
        # 早停判断逻辑
        if val_loss < best_val_loss:  # 如果验证损失改善
            best_val_loss = val_loss  # 更新最佳验证损失
            epochs_no_improve = 0     # 重置计数器
            # 保存最佳模型
            try:
                PATH = './output/cifar_model.pth'
                torch.save(model.state_dict(), PATH)
                print(f'最佳模型epoch{epoch+1}保存在：{PATH}')
            except Exception as e:
                print(f"保存失败: {str(e)}") # 异常处理
        else:  # 如果验证损失没有改善
            epochs_no_improve += 1  # 计数器增加
            if epochs_no_improve >= 10:  # 达到早停条件
                print(f'早停触发，最佳模型已保存至{PATH}')
                break  # 终止训练循环

        scheduler.step()
    print("训练结束")

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.show()