from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import FaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace
from evaluate_batch import eval_main, write_to_log

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--mode', default='train', help="['train', 'val', 'test']")
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default='./weights/mobilenet0.25_Final.pth', help='resume net for retraining')
# parser.add_argument('--resume_net', default='./weights/2023_06_25_03_53_49/mobilenet0.25_1.pth', help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')

args = parser.parse_args()
# initiate a weight folder and log file
run_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print("run_id:", run_id)
save_folder = args.save_folder + run_id + "/"
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
log_file = f"{save_folder}/train_{run_id}.log"
write_to_log(log_file, f"RUN_ID:{run_id}")

cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
print("train img_dim:", img_dim)
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']
early_stop_epoch = cfg['early_stop_epoch']
IOU_loss = cfg['iou_loss']
print("Use IOU_loss:", IOU_loss)

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
# training_dataset = args.training_dataset
mode = args.mode

data_folder_path = f"{cfg['data_folder']}/{mode}/"
# train_label_files = ['label_aurora.txt', 'label_driver.txt', 'label_consumer.txt', 'label_wider_face.txt', 'label_retina_fail.txt']
train_label_files = ['label_aurora.txt', 'label_driver.txt', 'label_consumer.txt', 'label_retina_fail.txt']
val_label_files = ['label_aurora.txt', 'label_driver.txt', 'label_consumer.txt', 'label_retina_fail.txt']


# data_folder_path = f"{cfg['data_folder']}/{mode}/"
# train_label_files = ['label_aurora.txt']
# val_label_files = ['label_aurora.txt']
# data_folder_path = "/home/jovyan/data/vol_3/face_detection_files/face_detection_v1_1/data/test_training/"
# train_label_files = ['label_aurora.txt', 'label_driver.txt', 'label_consumer.txt']
# val_label_files = ['label_test.txt']

net = RetinaFace(cfg=cfg)
# print("Printing net...")
# print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    print("Resume training from weight path:" +  args.resume_net)
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cpu()
    # net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cpu()
    # net = net.cuda()

cudnn.benchmark = True


optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False, IOU_loss=IOU_loss)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cpu()
    # priors = priors.cuda()

def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = FaceDetection( label_folder_path=data_folder_path,label_files=train_label_files,preproc=preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    best_ap = 0
    best_epoch = None
    prev_mAP = 0
    epoch_not_improving = 1
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        # images = images.cuda()
        images = images.cpu()
        targets = [anno.cpu() for anno in targets]
        # targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || Total Loss:{:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                      epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), loss.item() ,lr, batch_time, str(datetime.timedelta(seconds=eta))))
        if epoch % 1 == 0 and (iteration % epoch_size) + 1 == epoch_size:
            write_to_log(log_file, "[{}] epoch {} || Loc: {:.4f}, Cla: {:.4f}, Landm: {:.4f}, Total loss: {:.4f} || LR: {:.8f} || train_img_dim: {}".format(mode, epoch, loss_l.item(), loss_c.item(), loss_landm.item(),loss.item() ,lr, img_dim))
            # eval val performace
            torch.save(net.state_dict(), save_folder + cfg['name'] + f'_{epoch}.pth')
            ap = eval_main('val', f"{cfg['data_folder']}/val/", val_label_files, cfg, net, log_file_path=log_file, epoch=epoch)
            if ap > best_ap:
                best_ap = ap
                best_epoch = epoch
                print(f"Best Val mAP at epoch {epoch} with {ap}!")
                write_to_log(log_file, f"[{mode}] Current best epoch: {best_epoch}")

                torch.save(net.state_dict(), save_folder + cfg['name'] + f'_best.pth')
            else:
                print(f"No improvement in mAP:{ap}, current best map is {best_ap} at epoch {best_epoch}")
            # check early stopping
            if ap <= prev_mAP:
                epoch_not_improving += 1
            else:
                epoch_not_improving = 1
            if epoch_not_improving >= early_stop_epoch:
                break
            prev_mAP = ap
            net.train()
    write_to_log(log_file, f"[{mode}] Final best epoch: {best_epoch}")
    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
