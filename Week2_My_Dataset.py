import os
from skimage import io
import torch
from PIL import Image
import torchvision.datasets.mnist as mnist
from torch.utils.data.dataset import Dataset

root = './data/mnist/Mnist/raw'
train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
)

test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
)

print('train set:', train_set[0].size())
print('test set:', test_set[0].size())


def convert_to_img(train=True):
    if train:
        f = open(root + '/train.txt', 'w')
        data_path = root + '/train/'
        if not os.path.exists(data_path): os.makedirs(data_path)

        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            int_label = str(label).replace('tensor(', '')
            int_label = int_label.replace(')', '')
            f.write(img_path + ' ' + str(int_label) + '\n')
        f.close()
    else:
        f = open(root + '/test.txt', 'w')
        data_path = root + '/test/'
        if not os.path.exists(data_path): os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            int_label = str(label).replace('tensor(', '')
            int_label = int_label.replace(')', '')
            f.write(img_path + ' ' + str(int_label) + '\n')
        f.close()
#这个函数是生成训练测试路径用txt文件保存
convert_to_img(True)
convert_to_img(False)
class DataProcessingMnist(Dataset):
    """
    pytorch训练数据时需要数据集为Dataset类，便于迭代等等，
    这里将加载保存之后的数据封装成Dataset类，
    继承该类需要写初始化方法（__init__），获取指定下标数据的方法__getitem__），
    获取数据个数的方法（__len__）。
    这里尤其需要注意的是要把label转为LongTensor类型的。
    """
    def __init__(self,root_path,imgfile_path,labelfile_path,transform=None):
        self.root_path = root_path
        self.transform = transform
        img_file = open((root_path+imgfile_path),'r')
        self.image_name = [x.strip() for x in img_file]
        img_file.close()
        label_file = open((root_path+labelfile_path),'r')
        label = [int(x.strip()) for x in label_file]
        label_file.close()
        self.label = torch.LongTensor(label)#这句很重要，一定要把label转为LongTensor类型的

    def __getitem__(self,idx):
        image = Image.open(str(self.image_name[idx]))
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.label[idx]
        return image,label

    def __len__(self):
        return len(self.image_name)

#定义完自己的类之后测试一下。
root_path = 'data/'
training_path = 'trainingset/'
test_path = 'testset/'
training_imgfile = training_path + 'trainingset_img.txt'
training_labelfile = training_path + 'trainingset_label.txt'
dataset = DataProcessingMnist(root_path, training_imgfile, training_labelfile)
#图像路径
name = dataset.image_name
#获取固定下标的图像
im, label = dataset.__getitem__(0)

