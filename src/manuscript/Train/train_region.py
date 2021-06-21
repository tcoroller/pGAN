# pylint: disable=E0401, C0413 # E0401: import-error (torch, sklearn & cv2); C0413: import-position
# pylint: disable=W0511  # fix-me issue (todos)
# pylint: disable=E1101  # no-member (torch)
import os
import sys
from time import time
import mlflow
import torch

import torch.optim as optim
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import one_hot
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.insert(0, 'src')

# Architecture
from manuscript.Train.fixed_architecture import Generator, Discriminator

# Dataloading and preprocessing
from manuscript.Train.restricted.train_dataset import F2305Dataset, VU_PATH, spineNet_split
from manuscript.Train.restricted.test_dataset import A2209Dataset
from manuscript.Train.batchers import F2305Batcher, A2209Batcher
from manuscript.Train.transforms.transforms import Zoom, RotationFixedAxis, RandomCrop, Whitening

# Helpers and arg parser
from training_parser import parse_training_arguments
from helper import grid_image

# ref:
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py
# https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_gradient_penalty.py
# https://github.com/cyclomon/3dbraingen
# https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py

# relevant additional links:
# https://distill.pub/2016/deconv-checkerboard/
# https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html
#

# =================================================================================================================================
# parameters
parser = parse_training_arguments()
args = parser.parse_args()
no_log_keys = ["lr", "time_scale", "exp", "run_name", "continued", "checkpoint", "no_pbar", "n_step_eval",
               "eval_size"]

if args.lr is not None:
    args.lrG = args.lr
    args.lrD = args.time_scale * args.lr

args.gpu = torch.cuda.is_available()

# device
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# =================================================================================================================================
# PREPARING TRAIN SET
source_path = ("/modesim" if args.gpu else "") + VU_PATH

image_shape = (9, 64, 64)
image_list = spineNet_split()['train']

# Training dataset, source path is the location where the real vertebrae images are in .npy format
data = F2305Dataset(shape=image_shape, source_path=source_path)
# The data.prepare preprocesses the data to image shape and appends labels in a dataframe
# We only are using T1, (types has to be tuple or list so the "," is necessary)
# subset_fraction is the allows for automated val/train split, default is 100% of data as training
# image list is the list of images to be used as training
data.prepare(types=('T1',), subset_fraction=args.subset_fraction, image_list=image_list)

# batcher is the standard format for dataset in pytorch for iterating over the data

batcher = F2305Batcher(data.dataset, data.scan_path)
# can also add transforms there
if args.data_augm:
    batcher = F2305Batcher(data.dataset, data.scan_path,
                           transforms.Compose([Zoom(1.15, random=True),
                                               RotationFixedAxis(max_angle=6.0, axis=0, reshape=True),
                                               RandomCrop((9, 64, 64)), Whitening()]))

# creating loader,
loader = DataLoader(batcher, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

# =================================================================================================================================
# PREPARING TEST SET

# CREATING TEST SET WITH A2209
image_shape = (9, 64, 64)  # image shape for training

# The functions are mirrored from F2305 dataset/batcher class
testdata = A2209Dataset(shape=image_shape)
testdata.prepare(types=('T1',))
vu_loader_test = DataLoader(A2209Batcher(testdata.dataset, testdata.scan_path),
                            batch_size=16, shuffle=True, num_workers=args.num_workers)

def noise_gen(size, p_region=data.probs['region'], c_fill_region=None):
    '''Generate Z vector with classes and noise for Generator
    input:
        size: number of images to output
        p_region: probability of sample each region
        c_fill_region: hard coding classes (in a array with labels between 0-2 (no one hot))
    OUTPUTS Cr (classes region) + Z (Cr + noise)'''

    # if not hard input for class, sample from distribution
    if c_fill_region is None:
        cr = np.random.choice(3, size=size, p=p_region)
    # otherwise use c_fill_region as labels
    else:
        cr = np.ones(size) * c_fill_region

    # change type to tensor
    cr = torch.from_numpy(cr).to(device).long()
    # labels as onehot
    c_list = [one_hot(cr, num_classes=3).float()]

    # concat classes + gaussian noise
    z = torch.cat(c_list + [torch.randn(size=(size, args.feature_dim), device=device)], dim=1)
    # cr is returned twice, once alone and one within the z vector (first 3 dimensions if 3 regions)
    return cr, z


# =================================================================================================================================
# logs
log_path = f"logs/acgan/{args.exp}/{args.run_name}"
writer = SummaryWriter(log_dir=log_path)
ckpt = log_path + ".pt"  # default checkpoint

# Using a fix Z for most images to analyze progression from epoch to epoch and regions
# regions are 0 0 1 1 2 2, with noise z1 z2 z1 z2 z1 z2
# 6 images
n_show = 6
# casting labels
cr_fill = np.array([0, 0, 1, 1, 2, 2])

# intial noise
z_same_fix = noise_gen(n_show, c_fill_region=cr_fill)[-1]
# z3 and z4 are now z1 and z2
z_same_fix[2:4, 3:] = z_same_fix[:2, 3:]
# z5 z6 are now z1 and z2
z_same_fix[4:6, 3:] = z_same_fix[:2, 3:]

def log_metrics(metrics: dict, step):
    '''Adding scalar at every epoch for good x axis'''
    mlflow.log_metrics(metrics, step=step)
    for _k, _v in metrics.items():
        writer.add_scalar(_k, _v, global_step=step)


def log_images(step, zfixed=z_same_fix):  # https://pytorch.org/docs/stable/tensorboard.html
    '''Logging images, one line as fixed with fixed '''
    # sample real images
    im_real = next(iter(DataLoader(batcher, batch_size=n_show, shuffle=True)))['im'].to(device)

    # generate fake images
    im_fake_fixed = netG(zfixed)
    im_fake_random = netG(noise_gen(n_show, c_fill_region=cr_fill)[-1])

    # write images to tensorboard
    writer.add_image("samples/fake/fixed", grid_image(im_fake_fixed), step)
    writer.add_image("samples/fake/random", grid_image(im_fake_random), step)
    writer.add_image("samples/real", grid_image(im_real), step)


def state_dict(net):  # for saving model checkpoint (nn.DataParallel)
    try:
        return net.module.state_dict()
    except AttributeError:
        return net.state_dict()


# =================================================================================================================================
# model

netG = Generator(feature_dim=args.feature_dim, nc=image_shape[0],
                 net_size=args.net_size, batch_norm=args.G_batch)

netD = Discriminator(nc=image_shape[0], net_size=args.net_size, batch_norm=args.D_batch)
netD.to(device)
netG.to(device)

G_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in G_parameters])
print("Learnable parameters in Generator :", params)

D_parameters = filter(lambda p: p.requires_grad, netD.parameters())
params = sum([np.prod(p.size()) for p in D_parameters])
print("Learnable parameters in Discriminator :", params)

# load checkpoint
init_step = 0
if args.continued and args.checkpoint is None:
    args.checkpoint = ckpt
if args.checkpoint is not None:
    args.continued = True
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        netD.load_state_dict(checkpoint['netD'])
        netG.load_state_dict(checkpoint['netG'])
        init_step = checkpoint['step']
        print(f'==> Resuming from checkpoint at step {init_step}: \n\t{os.path.abspath(args.checkpoint)}')
    else:
        print('==> Checkpoint file not exist, starting from scratch')
end_step = init_step + args.n_epoch

# parallel GPUs
if args.gpu:
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

# optimizers
optim_D = optim.Adam(netD.parameters(), lr=args.lrD, betas=(0.5, 0.999))
optim_G = optim.Adam(netG.parameters(), lr=args.lrG, betas=(0.5, 0.999))

# criteria
d_criterion = nn.BCELoss()  # discriminator
cb_criterion = nn.NLLLoss(weight=torch.FloatTensor(data.weights['berlin']).to(device))  # classification: berlin
cr_criterion = nn.NLLLoss(weight=torch.FloatTensor(data.weights['region']).to(device))  # classification: region (C, T, L)


#  =================================================================================================================================
# Training
def train(step):
    # count is used to normalize metrics based of num of iteration and is incremented every iteration
    count = len(loader)

    # initializing metric names
    metrics = dict.fromkeys(['loss/dis', 'loss/d_cls_real_balanced', 'loss/d_cls_fake_balanced',
                             'loss/adv_G', 'loss/adv_D', 'nn_loss/D', 'nn_loss/G'], 0.)

    # tracker to train imbalanced G and D (for example 2 iter of D for 1 iter of G)
    iter_G = 0

    # initializing list of predictions for logging balanced accuracy
    list_d_fake_preds = np.array([])
    list_d_fake_gt = np.array([])
    list_d_real_preds = np.array([])
    list_d_real_gt = np.array([])

    # main training loop, sampling images with loader
    for batch in tqdm(loader, desc="train", disable=args.no_pbar):

        ###########
        # Train D #
        ###########
        for param in netD.parameters():
            param.requires_grad = True

        netD.zero_grad()

        # SAMPLING REAL IMAGE
        im_real = batch['im'].to(device)
        im_real = im_real.float()
        batch_size = im_real.shape[0]
        # real labels
        cr_real = batch['region'].to(device).long()
        # logits for loss
        d_real = torch.ones(batch_size, device=device)  # discriminate label (real=1 vs fake=0)

        # soft_d is the adv output, soft_cr is class output
        # running on real data
        soft_d_real, soft_cr_real = netD(im_real)

        # computing adv loss with soft_d and logits
        d_loss_real = d_criterion(soft_d_real, d_real)
        # computing class loss with soft_cr and labels
        cr_loss_real = cr_criterion(soft_cr_real, cr_real)
        # full loss + backprop
        netD_loss_real = d_loss_real + args.alpha_region * cr_loss_real
        netD_loss_real.backward()

        # for logging D balanced accuracy on real data
        list_d_real_preds = np.concatenate((list_d_fake_preds,
                                            soft_cr_real.argmax(axis=1).squeeze().cpu().numpy()))
        list_d_real_gt = np.concatenate((list_d_fake_gt, cr_real.squeeze().cpu().numpy()))

        # SAMPLING FAKE IMAGE
        # logits for computing loss
        d_fake = torch.zeros(batch_size, device=device)

        # generating Z vector
        cr_fake, z = noise_gen(size=batch_size)
        # generating fake image with generator
        im_fake = netG(z)

        # getting output of Discriminator, using detach to not propagate gradient in G
        soft_d_fake, soft_cr_fake = netD(im_fake.detach())
        # adv loss
        d_loss_fake = d_criterion(soft_d_fake, d_fake)
        # class loss
        cr_loss_fake = cr_criterion(soft_cr_fake, cr_fake)
        # full loss with alpha weight
        netD_loss_fake = d_loss_fake + args.alpha_region * cr_loss_fake
        # backpropagate and update
        netD_loss_fake.backward()
        optim_D.step()

        # for logging D balanced accuracy on fake data
        list_d_fake_preds = np.concatenate((list_d_fake_preds,
                                            soft_cr_fake.argmax(axis=1).squeeze().cpu().numpy()))
        list_d_fake_gt = np.concatenate((list_d_fake_gt, cr_fake.squeeze().cpu().numpy()))

        metrics['loss/dis'] += (d_loss_real + d_loss_fake).item() / count
        metrics['nn_loss/D'] += (netD_loss_real + netD_loss_fake).item() / count
        metrics['loss/adv_D'] += (d_loss_real + d_loss_fake).item() / count

        ###########
        # Train G #
        ###########

        # making sure nothing is updated in D during the training of G
        for param in netD.parameters():
            param.requires_grad = False

        # running imbalanced sampling of D and G (here can only training D more often)
        if iter_G % args.diter == 0:

            netG.zero_grad()
            # new sampling (recommended to resample before updating G and D)
            cr_fake, z = noise_gen(size=batch_size)
            # generating fake image
            im_fake = netG(z)

            # logits for loss (ones here)
            d_fake_G = d_real
            soft_d_fake, soft_cr_fake = netD(im_fake)
            # adv loss
            d_loss_fake = d_criterion(soft_d_fake, d_fake_G)
            # class loss
            cr_loss_fake = cr_criterion(soft_cr_fake, cr_fake)
            # loss backaward and update
            netG_loss = d_loss_fake + args.alpha_region * cr_loss_fake
            netG_loss.backward()
            optim_G.step()

            # log Generator metrics
            metrics['nn_loss/G'] += netG_loss.item() / count
            metrics['loss/adv_G'] += (d_loss_fake).item() / count
        iter_G += 1

    ###########
    #   Log   #
    ###########
    # computing balanced acc on real images
    metrics['loss/d_cls_real_balanced'] += balanced_accuracy_score(np.array(list_d_real_preds),
                                                                   np.array(list_d_real_gt))
    # computing balanced acc on fake images
    metrics['loss/d_cls_fake_balanced'] += balanced_accuracy_score(np.array(list_d_fake_preds),
                                                                   np.array(list_d_fake_gt))
    # log to tensorboard
    log_metrics(metrics, step=step)
    log_images(step=step)

def evaluate_A2209(step):
    '''Running evaluation on test set
    Classifier from discriminator on test set
    metrics:
        time of test (to make sure it doesn't take too long to run the evaluate)
        test accuracy
        test balanced accuracy'''

    # Preparing labels and prediction lists (still using a fixed batchsize to run on gpu)
    labs_region = torch.Tensor().to(device).long()
    preds_region = torch.Tensor().to(device).long()

    start_eval_time = time()
    # Running D with no gradient propagation
    with torch.no_grad():
        for _, sample_test in tqdm(enumerate(vu_loader_test)):
            sample_test['im'] = sample_test['im'].to(device)
            sample_test['region'] = sample_test['region'].to(device).long()
            _, soft_cr_real = netD(sample_test['im'])

            curr_pred_region = soft_cr_real.argmax(axis=1)
            labs_region = torch.cat((labs_region, sample_test['region']), 0)
            preds_region = torch.cat((preds_region, curr_pred_region), 0)
    end_eval_time = time()

    # compute balanced accuracy
    acc_balanced_region = balanced_accuracy_score(labs_region.cpu().numpy(), preds_region.cpu().numpy())

    # compute baseline accuracy
    overall_acc_region = (torch.sum((labs_region == preds_region)*1.0)/len(labs_region)).cpu().numpy()
    log_metrics({'test/time': end_eval_time - start_eval_time,
                 'test/acc_region': float(overall_acc_region),
                 'test/balanced_region': float(acc_balanced_region)}, step=step)


# =================================================================================================================================
mlflow.set_experiment("acgan-" + args.exp)
with mlflow.start_run(run_name=args.run_name):
    # log parameters
    params = vars(args).copy()
    for _k in no_log_keys:
        if _k in params:
            params.pop(_k)
    # params["init_step"] = init_step
    params["end_step"] = end_step
    mlflow.log_params(params)
    writer.add_text("parameters", str(params))
    writer.add_text("resume", "python " + " ".join(sys.argv) + (f" --checkpoint {os.path.abspath(ckpt)}" if args.checkpoint is None else ""))

    # training
    start_time = time()
    duration_epoch = 0.
    for epoch in tqdm(range(init_step, end_step), desc="epoch", disable=args.no_pbar):
        time0 = time()
        train(step=epoch)
        duration_epoch += (time() - time0)
        # save and evaluate every epoch
        if epoch % 1 == 0 or epoch == end_step - 1:
            evaluate_A2209(epoch)
        if epoch % 10 == 0 or epoch == end_step - 1:
            ckpt_epoch = log_path + '_' + str(epoch).zfill(3) + ".pt"
            print(f"Done!\n==> Saving checkpoint at step {end_step} to: \n\t{os.path.abspath(ckpt_epoch)}")
            torch.save({'netD': state_dict(netD), 'netG': state_dict(netG), 'step': end_step}, ckpt_epoch)

    mlflow.log_metric("duration", time() - start_time)
    mlflow.log_metric("duration_epoch", duration_epoch/args.n_epoch)

    # saving model
    mlflow.end_run()
