import os
def get_root_dir():
    return os.path.dirname(os.path.dirname(__file__))
import sys
sys.path.append(get_root_dir())
import torch
import argparse
import yaml
import time
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.augmentations_mm import get_train_augmentation_adapt, get_val_augmentation
from semseg.losses import get_loss
from semseg.models import *
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from val_mm import evaluate,train_pre
from semseg.datasets import Mydata_adapt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def main(cfg, gpu, save_dir):
    start = time.time()
    best_mIoU = 0.0
    best_iou_ = 0.0
    best_epoch = 0
    num_workers = 0  # 8
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    resume_path = cfg['MODEL']['RESUME']
    # gpus = int(os.environ['WORLD_SIZE'])
    gpus = 1

    traintransform = get_train_augmentation_adapt(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])

    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform, dataset_cfg['MODALS'])
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'])
    class_names = trainset.CLASSES

    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes, dataset_cfg['MODALS'])
    resume_checkpoint = None
    if os.path.isfile(resume_path):
        resume_checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        msg = model.load_state_dict(resume_checkpoint['model_state_dict'])
        # print(msg)
        logger.info(msg)
    elif model_cfg['ADAPT'] == True and not os.path.isfile(resume_path):
        checkpoint = torch.load(model_cfg['PRETRAINED'], map_location=torch.device('cpu'))
        msg = model.load_state_dict(checkpoint)
        print(msg)
    else:
        model.init_pretrained(model_cfg['PRETRAINED'])
    model = model.to(device)

    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE'] // gpus
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)

    start_epoch = 0
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, int((epochs + 1) * iters_per_epoch), sched_cfg['POWER'],
                              iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])

    if train_cfg['DDP']:
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        sampler_val = None
        model = DDP(model, device_ids=[gpu], output_device=0, find_unused_parameters=True)
    else:
        sampler = RandomSampler(trainset)
        sampler_val = None

    if resume_checkpoint:
        start_epoch = resume_checkpoint['epoch'] - 1
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        best_mIoU = resume_checkpoint['best_miou']
        best_iou_ = resume_checkpoint['best_iou_']

    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True,
                             pin_memory=False, sampler=sampler)
    valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=False,
                           sampler=sampler_val)

    scaler = GradScaler(enabled=train_cfg['AMP'])
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer = SummaryWriter(str(save_dir))
        logger.info('================== model complexity =====================')
        cal_flops(model, dataset_cfg['MODALS'], logger)
        logger.info('================== model structure =====================')
        logger.info(model)
        logger.info('================== training config =====================')
        logger.info(cfg)

    if train_cfg['PLCStratey'] == 'SkePoint_prompted_BufSAM':
        #### sam
        sam_checkpoint = "./checkpoints/pretrained/sam/sam_vit_b_01ec64.pth"  # sam_vit_l_0b3195.pth"  #sam_vit_h_4b8939.pth"
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        sam = sam.to(device)
        predictor = SamPredictor(sam)
    else:
        predictor = None

    ep = 50
    for epoch in range(start_epoch, epochs):
        model.train()
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False

        if train_cfg['DDP']: sampler.set_epoch(epoch)

        train_loss = 0.0
        lr = scheduler.get_lr()
        lr = sum(lr) / len(lr)
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch,
                    desc=f"Epoch: [{epoch + 1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        for iter, (sample, lbl, img, name) in pbar:
            optimizer.zero_grad(set_to_none=True)
            sample = [x.to(device) for x in sample]

            if epoch == 0:
                lbl = lbl.long() ## prepared preliminary segmentation labels predicted by source-domain model
            elif epoch >0 and epoch < ep:
                lbl =torch.zeros((lbl.shape[0],lbl.shape[1],lbl.shape[2]))
                prepath = str(save_dir) +'\\process\\pre\\'+str(epoch-1)+'\\'
                for ni in range(len(name)):
                    label = torch.tensor(cv2.imread(prepath+name[ni])[:, :, (2, 1, 0)].transpose(2, 0, 1))[0, ...].unsqueeze(0)
                    label = label.long()
                    lbl[ni,:,:] = label
            elif epoch >= ep:
                lbl = torch.zeros((lbl.shape[0], lbl.shape[1], lbl.shape[2]))
                prepath = str(save_dir) +'\\process\\pre_pl\\' + str(ep-1) + '\\'
                for ni in range(len(name)):
                    label = torch.tensor(cv2.imread(prepath + name[ni])[:, :, (2, 1, 0)].transpose(2, 0, 1))[
                        0, ...].unsqueeze(0)
                    label = label.long()
                    lbl[ni, :, :] = label
                lbl = lbl.long()

            lbl = lbl.to(device)

            with autocast(enabled=train_cfg['AMP']):
                pre,corrected_pl = model.generate_pl_pre(sample, predictor,img,lbl,epoch,ep,name,str(save_dir))
                loss = loss_fn(pre,corrected_pl)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            if lr <= 1e-8:
                lr = 1e-8  # minimum of lr

            train_loss += loss.item()

            pbar.set_description(
                f"Epoch: [{epoch + 1}/{epochs}] Iter: [{iter + 1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter + 1):.8f}")

        train_loss /= iter + 1
        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()

        if ((epoch + 1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch + 1) > train_cfg['EVAL_START']) or (
                epoch + 1) == epochs:
            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                acc, macc, _, _, ious, miou = evaluate(model, valloader, device)
                train_pre(model, trainloader, device,epoch)
                iou_ = ious[-1]
                writer.add_scalar('val/mIoU', miou, epoch)
                writer.add_scalar('val/iou_', iou_, epoch)

                if iou_ > best_iou_:
                    prev_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    prev_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    if os.path.isfile(prev_best): os.remove(prev_best)
                    if os.path.isfile(prev_best_ckp): os.remove(prev_best_ckp)
                    best_mIoU = miou
                    best_iou_ = iou_
                    best_epoch = epoch + 1
                    cur_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    cur_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), cur_best)
                    # ---
                    torch.save({'epoch': best_epoch,
                                'model_state_dict': model.module.state_dict() if train_cfg[
                                    'DDP'] else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss,
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_miou': best_mIoU,
                                'best_iou_': best_iou_,
                                }, cur_best_ckp)
                    logger.info(print_iou(epoch, ious, miou, acc, macc, class_names))
                logger.info(f"Current epoch:{epoch} mIoU: {miou} Best mIoU: {best_mIoU} current_iou_: {iou_}")
        if epoch == epochs - 1:
            last_epoch = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_last_epoch.pth"

            torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), last_epoch)

    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    logger.info(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='../configs/rv_UDASS.yaml',
                        help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(12345)
    setup_cudnn()
    # gpu = setup_ddp()
    gpu = 0
    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    model = cfg['MODEL']['BACKBONE']
    exp_name = '_'.join([cfg['DATASET']['NAME'], model, modals])
    save_dir = Path(cfg['SAVE_DIR'], exp_name)
    if os.path.isfile(cfg['MODEL']['RESUME']):
        save_dir = Path(os.path.dirname(cfg['MODEL']['RESUME']))
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(save_dir / 'train.log')
    main(cfg, gpu, save_dir)
    cleanup_ddp()