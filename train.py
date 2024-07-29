import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import cv2
import os
import numpy as np
from time import time
from networks.dinknet import DinkNet34

ROOT = r'dataset_yuantu/2'
NAME = 'dlink34'
BATCHSIZE_PER_CARD = 5
LR = 2e-4
EPOCH = 500


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        if i == 0 and j == 0:
            score = torch.tensor(0.0, dtype=torch.float32)
        else:
            score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        sdc = self.soft_dice_coeff(y_true, y_pred)
        if sdc == 0:
            loss = 0
        else:
            loss = 1 - sdc
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask


def default_loader(id, root):
    img = cv2.imread(os.path.join(root, 'temp/' + str(id) + '_image.png'))
    mask = cv2.imread(os.path.join(root, 'temp/' + str(id) + '_label.png'), cv2.IMREAD_GRAYSCALE)

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask


class ImageFolder(data.Dataset):

    def __init__(self, trainlist, root):
        self.ids = trainlist
        self.loader = default_loader
        self.root = root

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        if img.size()[1] != mask.size()[2]:
            mask = mask.transpose(1, 2)
        return img, mask

    def __len__(self):
        return len(list(self.ids))


class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode=False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img(self, img):
        pred = self.net.forward(img)

        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask

    def test_batch(self):
        self.forward(volatile=True)
        mask = self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())

        mask = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.data.item()

    def TTA(self, img):
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).cuda())
        maska = self.net.forward(img5)
        maska = maska.squeeze().cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()
        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]
        mask3[mask3 <= 0.4] = 0
        mask3[mask3 > 0.4] = 1
        pred = np.asarray(mask3, dtype=np.uint8)
        return pred

    def optimize_test(self, metric):
        # self.forward()
        pred = self.TTA(self.img)
        self.mask = cv2.threshold(self.mask, 100, 1, cv2.THRESH_BINARY)[1]
        hist = metric.addBatch(pred, self.mask)
        cpa = metric.classPixelAccuracy()
        IoU = metric.IntersectionOverUnion()
        self.mask = torch.Tensor(self.mask)
        pred = torch.Tensor(pred)
        loss = self.loss(self.mask, pred)
        loss = loss.data.item()
        return loss, cpa[1], IoU[1]

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f \n' % (self.old_lr, new_lr))
        self.old_lr = new_lr


solver = MyFrame(DinkNet34, dice_bce_loss, LR)
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
imagelist = os.listdir(os.path.join(ROOT, 'image'))
if os.path.exists(os.path.join(ROOT, 'temp')):
    for f in os.listdir(os.path.join(ROOT, 'temp')):
        os.remove(os.path.join(ROOT, 'temp', f))
    os.removedirs(os.path.join(ROOT, 'temp'))
os.makedirs(os.path.join(ROOT, 'temp'))
print('正在预处理数据集')
for i in imagelist:
    for p in ['image', 'label']:
        img = cv2.imread(os.path.join(ROOT, p, i))
        if img is None:
            changetype = {'png': 'jpg', 'jpg': 'png'}
            img = cv2.imread(os.path.join(ROOT, p, i[:-3] + changetype[i[-3:]]))
        h, w, c = img.shape
        for r in range(h // 1024):
            r1024 = r * 1024
            for c in range(w // 1024):
                c1024 = c * 1024
                image = img[r1024: r1024 + 1024, c1024: c1024 + 1024, :]
                cv2.imwrite(os.path.join(ROOT, 'temp', i[:-4] + '_' + str(r) + str(c) + '_' + p + '.png'), image,
                            [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

trainlist = ['_'.join(x.split('_')[:-1]) for x in os.listdir(os.path.join(ROOT, 'temp')) if 'image' in x]
dataset = ImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0)

tic = time()
no_optim = 0
total_epoch = EPOCH
train_epoch_best_loss = 2.0

if not os.path.exists('weights/'):
    os.makedirs('weights/')

print('开始训练')
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0

    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader)
    print('********')
    print('epoch:' + str(epoch) + '    time:' + str(int(time() - tic)))
    print('train_loss:' + str(train_epoch_loss))

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/' + NAME + '_' + str(total_epoch) + '.th')
        print(".th has saved!")
    '''
    if no_optim > 6:
        mylog.write('early stop at %d epoch' % epoch)
        print('early stop at %d epoch' % epoch)
        break
    '''
    if no_optim > 7:
        # if solver.old_lr < 5e-7:
        #    break
        solver.load('weights/' + NAME + '_' + str(total_epoch) + '.th')
        solver.update_lr(1.2, factor=True)
        no_optim = 0
print('Finish!')
