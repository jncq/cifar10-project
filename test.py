import torch
from cifar10project import Model
import torchvision
import torchvision.transforms as transforms

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 封装一组转换函数对象作为转换器
transform = transforms.Compose(  # Compose是transforms的组合类
    [transforms.ToTensor(),  # ToTensor()类把PIL Image格式的图片和Numpy数组转换成张量
     transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))]  # 用均值和标准差归一化张量图像
)

# 声明批量大小，一批64张图片
batch_size = 64

# 实例化测试集
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        transform=transform, download=False)

# 实例化测试集加载器
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# CIFAR10数据集所有类别名称
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    PATH = './output/cifar_model.pth'
    # 加载模型参数,实例化模型对象
    model = Model().to(device)
    state_dict = torch.load(PATH, map_location=device)
    if any(key.startswith('module.') for key in state_dict):  # 更高效的检查
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model.eval()  
    correct = 0  # 预测正确的数量
    total = 0  # 测试集的总数

    # 由于不是训练，不需要计算输出的梯度
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # 模型输出预测结果
            outputs = model(images)

            # 选择置信度最高的类别作为预测类别
            _,predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # 打印准确率
    print(f'测试准确率: {100 * correct / total:.2f}%')