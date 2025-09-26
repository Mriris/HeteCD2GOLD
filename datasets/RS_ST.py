import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform
import matplotlib.pyplot as plt
from skimage.transform import rescale
from torchvision.transforms import functional as F
import random
from PIL import Image
from PIL import ImageFilter
from torchvision import transforms
# from osgeo import gdal_array
import cv2
from skimage import io, img_as_float
from scipy.ndimage import uniform_filter


MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A  = np.array([48.30,  46.27,  48.14])
MEAN_B = np.array([111.07, 111.07, 111.07])
STD_B  = np.array([49.41,  49.41,  49.41])

root = '/data/jingwei/yantingxuan/Datasets/CityCN/Split43'

class MultiImgPhotoMetricDistortion:
    """
    多图像一致的光度扰动。

    - 为一次样本内的多幅图像（如 A、B、C）采样同一组随机参数；
    - 仅对指定索引的图像执行颜色域变换（默认对光学 A/C 生效，跳过 SAR B）。

    参数说明（与 torchvision 定义一致）：
    - brightness: 亮度扰动幅度，取值 b 表示在 [1-b, 1+b] 中采样因子
    - contrast:   对比度扰动幅度，取值 c 表示在 [1-c, 1+c] 中采样因子
    - saturation: 饱和度扰动幅度，取值 s 表示在 [1-s, 1+s] 中采样因子
    - hue:        色调扰动幅度，取值 h 表示在 [-h, h] 中采样（h ∈ [0, 0.5]）
    - apply_to_indices: 需要应用的图像索引元组/列表
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.1,
        apply_to_indices=(0, 2),
    ) -> None:
        self.brightness = float(max(0.0, brightness))
        self.contrast = float(max(0.0, contrast))
        self.saturation = float(max(0.0, saturation))
        # hue 上限为 0.5（torchvision 约束）
        self.hue = float(min(max(0.0, hue), 0.5))
        if isinstance(apply_to_indices, (list, tuple)):
            self.apply_to_indices = tuple(apply_to_indices)
        else:
            self.apply_to_indices = (0, 2)

    def _sample_factors(self):
        # 采样一组因子并随机排列处理顺序
        factors = {}
        if self.brightness > 0:
            factors['brightness'] = 1.0 + (2.0 * random.random() - 1.0) * self.brightness
        if self.contrast > 0:
            factors['contrast'] = 1.0 + (2.0 * random.random() - 1.0) * self.contrast
        if self.saturation > 0:
            factors['saturation'] = 1.0 + (2.0 * random.random() - 1.0) * self.saturation
        if self.hue > 0:
            factors['hue'] = (2.0 * random.random() - 1.0) * self.hue

        ops = ['brightness', 'contrast', 'saturation', 'hue']
        random.shuffle(ops)
        return factors, ops

    def _apply_one(self, img: Image.Image, factors, ops) -> Image.Image:
        out = img
        for op in ops:
            if op not in factors:
                continue
            if op == 'brightness':
                out = F.adjust_brightness(out, factors['brightness'])
            elif op == 'contrast':
                out = F.adjust_contrast(out, factors['contrast'])
            elif op == 'saturation':
                # 仅对 RGB 图像有效
                if out.mode in ('RGB', 'RGBA'):
                    out = F.adjust_saturation(out, factors['saturation'])
            elif op == 'hue':
                if out.mode in ('RGB', 'RGBA'):
                    out = F.adjust_hue(out, factors['hue'])
        return out

    def __call__(self, imgs):
        if not isinstance(imgs, (list, tuple)) or len(imgs) == 0:
            return imgs
        factors, ops = self._sample_factors()
        out_imgs = []
        for idx, img in enumerate(imgs):
            if idx in self.apply_to_indices:
                out_imgs.append(self._apply_one(img, factors, ops))
            else:
                out_imgs.append(img)
        return out_imgs

class CDDataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot=False,
            with_random_crop=False,
            with_scale_random_crop=False,
            with_random_blur=False,
            random_color_tf=False,
            with_multi_img_photometric=False
    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
        self.random_color_tf = random_color_tf
        self.with_multi_img_photometric = with_multi_img_photometric
        self.photometric = (
            MultiImgPhotoMetricDistortion(apply_to_indices=(0, 2))
            if with_multi_img_photometric else None
        )
    def transform(self, imgs, labels, to_tensor=True):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        # resize image and covert to tensor
        imgs = [F.to_pil_image(img) for img in imgs]
        if self.img_size is None:
            self.img_size = None

        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size, self.img_size):
                imgs = [F.resize(img, [self.img_size, self.img_size], interpolation=3)
                        for img in imgs]
        else:
            self.img_size = imgs[0].size[0]

        labels = [F.to_pil_image(img) for img in labels]
        if len(labels) != 0:
            if labels[0].size != (self.img_size, self.img_size):
                labels = [F.resize(img, [self.img_size, self.img_size], interpolation=0)
                        for img in labels]

        random_base = 0.5
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [F.hflip(img) for img in imgs]
            labels = [F.hflip(img) for img in labels]

        if self.with_random_vflip and random.random() > 0.5:
            imgs = [F.vflip(img) for img in imgs]
            labels = [F.vflip(img) for img in labels]

        if self.with_random_rot and random.random() > random_base:
            angles = [90, 180, 270]
            index = random.randint(0, 2)
            angle = angles[index]
            imgs = [F.rotate(img, angle) for img in imgs]
            labels = [F.rotate(img, angle) for img in labels]

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=imgs[0], scale=(0.8, 1.0), ratio=(1, 1))

            imgs = [F.resized_crop(img, i, j, h, w,
                                    size=(self.img_size, self.img_size),
                                    interpolation=Image.CUBIC)
                    for img in imgs]

            labels = [F.resized_crop(img, i, j, h, w,
                                      size=(self.img_size, self.img_size),
                                      interpolation=Image.NEAREST)
                      for img in labels]

        if self.with_scale_random_crop and random.random() > 0.5:
            # rescale
            scale_range = [1, 1.2]
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            labels = [pil_rescale(img, target_scale, order=0) for img in labels]
            # crop
            # 统一使用PIL的 (width, height) 约定
            imgsize = imgs[0].size  # (w, h)
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
                    for img in imgs]
            labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255)
                    for img in labels]

        if self.with_random_blur and random.random() > 0.5:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                    for img in imgs]

        # 多图像一致的光度扰动（默认作用于 A/C）
        if self.with_multi_img_photometric and random.random() > 0.5:
            imgs = self.photometric(imgs)

        if self.random_color_tf and random.random() > 0.5:
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
            # 仅对 A/C 执行颜色扰动，跳过 B（SAR）
            imgs_tf = []
            for idx, img in enumerate(imgs):
                if idx in (0, 2):  # A 和 C
                    tf = transforms.ColorJitter(
                                color_jitter.brightness, 
                                color_jitter.contrast, 
                                color_jitter.saturation,
                                color_jitter.hue)
                    imgs_tf.append(tf(img))
                else:
                    imgs_tf.append(img)
            imgs = imgs_tf
            
        # if to_tensor:
        #     # to tensor
        #     imgs = [TF.to_tensor(img) for img in imgs]
        labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
                for img in labels]
            
        #     imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #             for img in imgs]
        # img1 = np.array(imgs[0])
        imgs = [F.to_tensor(img) for img in imgs]
        return imgs, labels


def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype)*default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    w, h = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw


def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    width, height = img.size
    target_size = (int(np.round(width*scale)), int(np.round(height*scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size == img.size:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size, resample)

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0
def lee_filter(img, size):
    img = img_as_float(img)  # 转换图像到浮点型，范围在0到1
    mean_img = uniform_filter(img, size)  # 计算局部均值
    mean_sqr_img = uniform_filter(img**2, size)  # 计算局部平方均值
    var_img = mean_sqr_img - mean_img**2  # 计算局部方差
    noise = np.mean(var_img)
    coeff = var_img / (var_img + noise)
    result_img = mean_img + coeff * (img - mean_img)
    return result_img
def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    im = im/255
    return im

def read_RSimages(mode, rescale=False):
    #assert mode in ['train', 'val', 'train_unchange']
    img_A_dir = os.path.join(root, mode, 'A')  # 时间点1光学图像
    img_B_dir = os.path.join(root, mode, 'B')  # 时间点2 SAR图像
    img_C_dir = os.path.join(root, mode, 'C')  # 时间点2光学图像（如果不存在则报错）
    label_A_dir = os.path.join(root, mode, 'label')
    
    data_list = os.listdir(img_A_dir)
    imgs_list_A, imgs_list_B, imgs_list_C, labels_A = [], [], [], []
    count = 0
    
    # 检查C目录是否存在，如果不存在则报错
    if not os.path.exists(img_C_dir):
        raise ValueError("C目录不存在")
    
    for it in data_list:
        img_A_path = os.path.join(img_A_dir, it)
        img_B_path = os.path.join(img_B_dir, it)
        img_C_path = os.path.join(img_C_dir, it)
        label_A_path = os.path.join(label_A_dir, it)
        imgs_list_A.append(img_A_path)
        imgs_list_B.append(img_B_path)
        imgs_list_C.append(img_C_path)
        label_A = io.imread(label_A_path)
        labels_A.append(label_A)
        count+=1
        if not count%500: print('%d/%d images loaded.'%(count, len(data_list)))
    
    print(labels_A[0].shape)
    print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')
    
    return imgs_list_A, imgs_list_B, imgs_list_C, labels_A

class Data(data.Dataset):
    def __init__(self, mode, random_flip = False, use_multi_img_photometric: bool = False):
        self.random_flip = random_flip
        self.imgs_list_A, self.imgs_list_B, self.imgs_list_C, self.labels = read_RSimages(mode)
        self.mode = mode
        self.augm = CDDataAugmentation(
                    img_size=512,
                    with_random_hflip=True,
                    with_random_vflip=True,
                    with_scale_random_crop=True,
                    with_random_blur=True,
                    random_color_tf=(not use_multi_img_photometric),
                    with_multi_img_photometric=use_multi_img_photometric
                )
    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_A = io.imread(self.imgs_list_A[idx])  # 时间点1光学图像
        img_B = io.imread(self.imgs_list_B[idx])  # 时间点2 SAR图像
        img_C = io.imread(self.imgs_list_C[idx])  # 时间点2光学图像
        name = self.imgs_list_A[idx].split('/')[-1].replace('.tif','.png')
        label= self.labels[idx]//255
        
        if self.mode=="train":
            [img_A, img_B, img_C], [label] = self.augm.transform([img_A, img_B, img_C], [label])
        else:
            img_A = F.to_tensor(img_A)
            img_B = F.to_tensor(img_B)
            img_C = F.to_tensor(img_C)
            label = torch.from_numpy(np.array(label, np.uint8)).unsqueeze(dim=0)
        
        return img_A, img_B, img_C, label.squeeze(), name

    def __len__(self):
        return len(self.imgs_list_A)

class Data_test(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_A = []
        self.imgs_B = []
        self.mask_name_list = []
        imgA_dir = os.path.join(test_dir, 'pre')
        imgB_dir = os.path.join(test_dir, 'post')
        data_list = os.listdir(imgA_dir)
        for it in data_list:
            if (it[-4:]=='.png'):
                img_A_path = os.path.join(imgA_dir, it)
                img_B_path = os.path.join(imgB_dir, it)
                self.imgs_A.append(io.imread(img_A_path))
                self.imgs_B.append(io.imread(img_B_path))
                self.mask_name_list.append(it)
        self.len = len(self.imgs_A)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        img_A = self.imgs_A[idx]
        img_B = self.imgs_B[idx]
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A), F.to_tensor(img_B)

    def __len__(self):
        return self.len

