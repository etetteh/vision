from __future__ import print_function, division

import torchvision, torchvision.transforms
import sklearn, sklearn.model_selection

import os
import copy
import time
import random
import argparse
import numpy as np
from functools import partial

import timm
import torch, torchvision

from torch import nn
from timm.loss import BinaryCrossEntropy, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from torchvision import datasets, models
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

from torch.utils.data import random_split
from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import transforms as T
import utils

import torchxrayvision as xrv
from torchxrayvision.datasets import Dataset

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, pb2, AsyncHyperBandScheduler



from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)


def create_splits(dataset, seed):
    if "patientid" not in dataset.csv:
        dataset.csv["patientid"] = ["{}-{}".format(dataset.__class__.__name__, i) for i in range(len(dataset))]
        
    gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=seed)
    
    traininds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
    traindataset = xrv.datasets.SubsetDataset(dataset, traininds)
    test_dataset = xrv.datasets.SubsetDataset(dataset, test_inds)
    
    train_inds, valid_inds = next(gss.split(X=range(len(traindataset)), groups=traindataset.csv.patientid))
    train_dataset = xrv.datasets.SubsetDataset(traindataset, train_inds)
    valid_dataset = xrv.datasets.SubsetDataset(traindataset, valid_inds)

    num_classes = len(dataset.pathologies)
    return train_dataset, valid_dataset, test_dataset, num_classes


def load_data(config, data_dir='./data'):
    
    ### Load NIH Dataset ###
    if "nih" in cfg.dataset:
        dataset = xrv.datasets.NIH_Dataset(
            imgpath=cfg.dataset_dir + "/images-224-NIH", 
            csvpath=cfg.dataset_dir + "/Data_Entry_2017_v2020.csv.gz",
            bbox_list_path=cfg.dataset_dir + "/BBox_List_2017.csv.gz",
            transform=None, 
            data_aug=None,
            unique_patients=False
        )

    ### Load CHEXPERT Dataset ###
    elif "chex" in cfg.dataset:
        dataset = xrv.datasets.CheX_Dataset(
            imgpath=cfg.dataset_dir + "/CheXpert-v1.0-small",
            csvpath=cfg.dataset_dir + "/CheXpert-v1.0-small/train.csv",
            transform=None,
            data_aug=None,
            unique_patients=False
        )

    ### Load MIMIC_CH Dataset ###
    elif "mimic" in cfg.dataset:
        dataset = xrv.datasets.MIMIC_Dataset(
            imgpath=cfg.dataset_dir + "/images-224-MIMIC/files",
            csvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
            metacsvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
            transform=None,
            data_aug=None,
            unique_patients=False
        )

    ### Load PADCHEST Dataset ###
    elif "pc" in cfg.dataset:
        dataset = xrv.datasets.PC_Dataset(
            imgpath=cfg.dataset_dir + "/PC/images-224",
            csvpath=cfg.dataset_dir + "/PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
            transform=None,
            data_aug=None,
            unique_patients=False
        )

    elif "google" in cfg.dataset:
        dataset = xrv.datasets.NIH_Google_Dataset(
            imgpath=cfg.dataset_dir + "/images-224-NIH",
            csvpath=cfg.dataset_dir + "/google2019_nih-chest-xray-labels.csv.gz", 
            transform=None, 
            data_aug=None
        )
        
    elif "rsna" in cfg.dataset:
        dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
                imgpath=cfg.dataset_dir + "/kaggle-pneumonia-jpg/stage_2_train_images_jpg",
                csvpath=cfg.dataset_dir + "/kaggle-pneumonia-jpg/stage_2_train_labels.csv",
                dicomcsvpath=cfg.dataset_dir + "kaggle_stage_2_train_images_dicom_headers.csv.gz",
                transform=None,
                data_aug=None,
                unique_patients=False
        )
        
    elif "openi" in cfg.dataset:
        OPENI_dataset = xrv.datasets.Openi_Dataset(
                imgpath=cfg.dataset_dir + "/OpenI/images/",
                xmlpath=cfg.dataset_dir + "NLMCXR_reports.tgz", 
                dicomcsv_path=cfg.dataset_dir + "nlmcxr_dicom_metadata.csv.gz",
                tsnepacsv_path=cfg.dataset_dir + "nlmcxr_tsne_pa.csv.gz",
                transform=None,
                data_aug=None
        )
    else:
        raise Exception("The specified Dataset is not considered in this work. Choose one of ['nih', 'chex', 'mimic', 'pc', 'google', 'rsna', 'openi']")
    #############################################################################    
    augment = [
               xrv.datasets.ToPILImage(),
               transforms.Grayscale(num_output_channels=3),
               transforms.RandomHorizontalFlip()
               ]

    ######### Recipe 2 #########           
    if cfg.random:
        augment.append(autoaugment.RandAugment(num_ops=config["num_ops"], magnitude=config["magnitude"], num_magnitude_bins=config["num_magnitude_bins"], interpolation=config["interpolation"]))
    elif cfg.num_ops:
        augment.append(autoaugment.RandAugment(num_ops=cfg.num_ops, magnitude=cfg.magnitude, num_magnitude_bins=cfg.num_magnitude_bins, interpolation=InterpolationMode(cfg.interpolation)))

    if cfg.trivial:
        augment.append(autoaugment.TrivialAugmentWide(num_magnitude_bins=config["num_magnitude_bins"], interpolation=config["interpolation"]))
    elif cfg.num_magnitude_bins:
        augment.append(autoaugment.TrivialAugmentWide(num_magnitude_bins=cfg.num_magnitude_bins, interpolation=InterpolationMode(cfg.interpolation)))

    augment.extend([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            ])  
    ######### Recipe 4 #########
    if cfg.random_erase:
        augment.append(transforms.RandomErasing(p=config["random_erase_prob"]))
    elif cfg.random_erase_prob:
        augment.append(transforms.RandomErasing(p=cfg.random_erase_prob))

    augment = transforms.Compose(augment)

    test_augment = transforms.Compose([
                xrv.datasets.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                ])

    data_transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])

    train_dataset, valid_dataset, test_dataset, num_classes = create_splits(dataset, seed=cfg.seed)

    train_dataset.transform, train_dataset.data_aug = data_transform, augment
    valid_dataset.transform, valid_dataset.data_aug = data_transform, test_augment
    test_dataset.transform, test_dataset.data_aug = data_transform, test_augment

    # print(f"Train transform: {train_dataset.transform}")
    return train_dataset, valid_dataset, test_dataset, num_classes



def train_epoch(config, epoch, model, ema_model, device, data_loader, optimizer, criterion, limit=None):
    model.train()

    weights = np.nansum(data_loader.dataset.labels, axis=0)
    weights = weights.max() - weights + weights.mean()
    weights = weights/weights.max()
    weights = torch.from_numpy(weights).to(device).float()
    # print("task weights", weights)
    
    avg_loss = []
    # t = tqdm(train_loader)
    for batch_idx, samples in enumerate(data_loader, 0):
        
        if limit and (batch_idx > limit):
            print("breaking out")
            break
            
        optimizer.zero_grad()
        
        images = samples["img"].float().to(device)
        targets = samples["lab"].to(device)

        outputs = model(images)
        
        loss = torch.zeros(1).to(device).float()
        for task in range(targets.shape[1]):
            task_output = outputs[:,task]
            task_target = targets[:,task]
            mask = ~torch.isnan(task_target)
            task_output = task_output[mask]
            task_target = task_target[mask]
            if len(task_target) > 0:
                task_loss = criterion(task_output.float(), task_target.float())
                loss += weights[task]*task_loss
                
        loss = loss.sum()        
        loss.backward()

        avg_loss.append(loss.detach().cpu().numpy())
        
        if batch_idx % 2000 == 1999:
            print(f"Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}")

        optimizer.step()
        
        if cfg.model_ema_steps:
            if ema_model and i % cfg.model_ema_steps == 0:
                ema_model.update_parameters(model)
                if epoch < cfg.lr_warmup_epochs:
                    ema_model.n_averaged.fill_(0)
        else:           
            if ema_model and batch_idx % config["model_ema_steps"] == 0:
                ema_model.update_parameters(model)
                if epoch < config["lr_warmup_epochs"]:
                    ema_model.n_averaged.fill_(0)


def valid_test_epoch(name, epoch, model, device, data_loader, criterion, limit=None):
    model.eval()

    avg_loss = []
    task_outputs={}
    task_targets={}
    for task in range(data_loader.dataset[0]["lab"].shape[0]):
        task_outputs[task] = []
        task_targets[task] = []
        
    with torch.no_grad():
        # t = tqdm(data_loader)
        for batch_idx, samples in enumerate(data_loader, 0):

            if limit and (batch_idx >= limit):
                print("breaking out")
                break
            
            images = samples["img"].to(device)#.cuda()
            targets = samples["lab"].to(device)#.cuda()

            outputs = model(images)
            
            loss = torch.zeros(1).to(device).double()
            for task in range(targets.shape[1]):
                task_output = outputs[:,task]
                task_target = targets[:,task]
                mask = ~torch.isnan(task_target)
                task_output = task_output[mask]
                task_target = task_target[mask]
                if len(task_target) > 0:
                    loss += criterion(task_output.double(), task_target.double())
                
                task_outputs[task].append(task_output.detach().cpu().numpy())
                task_targets[task].append(task_target.detach().cpu().numpy())

            loss = loss.sum()
            avg_loss.append(loss.detach().cpu().numpy())
            
        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])
    
        task_aucs = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task]))> 1:
                task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])

    return auc, np.mean(avg_loss), task_aucs



def train(config, checkpoint_dir=None, data_dir=None):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    train_dataset, valid_dataset, _, num_classes = load_data(config, data_dir=data_dir)
    
    if "deit" in cfg.model:
        model = torch.hub.load("facebookresearch/deit:main", cfg.model+"_patch16_224", pretrained=True)
        if cfg.fext:
            print(f"Performing feature extraction with {cfg.model} model")
            for param in model.parameters():
                param.requires_grad = False
        out_chn = model.patch_embed.proj.out_channels
        model.patch_embed.proj = torch.nn.Conv2d(1, out_chn, kernel_size=(16, 16), stride=(16, 16))
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes)
    elif "resnet" in cfg.model:
        model = torchvision.models.__dict__[cfg.model](pretrained=True)
        if cfg.fext:
            print(f"Performing feature extraction with {cfg.model} model")
            for param in model.parameters():
                param.requires_grad = False
        out_chn = model.conv1.out_channels
        model.conv1 = torch.nn.Conv2d(1, out_chn, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes, bias=True)
    else:
        raise Exception("The specified model isn't considered for this work. Choose any ResNet or DEIT model.")
        
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    
    ######### Recipe 10 #########
    ema_model = None
    if cfg.ema:
        adjust = 1 * cfg.batch_size * config["model_ema_steps"] / cfg.epochs
        alpha = 1.0 - config["model_ema_decay"]
        alpha = min(1.0, alpha * adjust)
        ema_model = utils.ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)
    elif (cfg.model_ema_decay and cfg.model_ema_steps):
        adjust = 1 * cfg.batch_size * cfg.model_ema_steps / cfg.epochs
        alpha = 1.0 - cfg.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        ema_model = utils.ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)
        
    if "deit" in cfg.model:
        if cfg.fext:
            parameters = model.head.parameters()
        else:
            parameters = model.parameters()
    elif "resnet" in cfg.model:
        if cfg.fext:
            parameters = model.fc.parameters()
        ese:
            parameters = model.parameters()
        
    
    ######### Recipe 8 #########                   
    if cfg.wd_tune:
        norm_weight_decay=0.0
        param_groups = utils.split_normalization_params(model)
        wd_groups = [norm_weight_decay, config["weight_decay"]]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]
    elif cfg.weight_decay:
        norm_weight_decay=0.0
        param_groups = utils.split_normalization_params(model)
        wd_groups = [norm_weight_decay, cfg.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]
    
    
    ######### Recipe 1 #########
    if cfg.lr_optim:
        optimizer = torch.optim.AdamW(parameters, lr=config["lr"])
        main_lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs - config["lr_warmup_epochs"])
        warmup_lr_scheduler = LinearLR(optimizer, start_factor=config["lr_warmup_decay"], total_iters=config["lr_warmup_epochs"])
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[config["lr_warmup_epochs"]])
    elif cfg.lr:
        optimizer = torch.optim.AdamW(parameters, lr=cfg.lr)
        main_lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.lr_warmup_epochs)
        warmup_lr_scheduler = LinearLR(optimizer, start_factor=cfg.lr_warmup_decay, total_iters=cfg.lr_warmup_epochs)
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[cfg.lr_warmup_epochs])
        
    ######### Recipe 5 #########
    if cfg.mixup or cfg.cutmix:
        train_criterion = BinaryCrossEntropy(smoothing=0.0)
    elif cfg.label_smoothing:
        if cfg.smooth:
            train_criterion = BinaryCrossEntropy(smoothing=cfg.smooth)
        else:
            train_criterion = BinaryCrossEntropy(smoothing=config["label_smoothing"])
    else:
        train_criterion = BinaryCrossEntropy(smoothing=0.0)
    train_criterion = train_criterion.to(device)

    valid_criterion = BinaryCrossEntropy(smoothing=0.0).to(device)
    
    collate_fn = None
    mixup_transforms = []
    ######### Recipe 6 #########
    if cfg.mixup:
        mixup_transforms.append(T.RandomMixup(num_classes, p=1.0, alpha=config["mixup_alpha"]))
    elif cfg.mixup_alpha:
        mixup_transforms.append(T.RandomMixup(num_classes, p=1.0, alpha=mixup_alpha))
    ######### Recipe 7 #########    
    if cfg.cutmix:
        mixup_transforms.append(T.RandomCutmix(num_classes, p=1.0, alpha=config["cutmix_alpha"]))
    elif cfg.cutmix_alpha:
        mixup_transforms.append(T.RandomCutmix(num_classes, p=1.0, alpha=cfg.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*default_collate(batch))
        
    if checkpoint_dir:
        checkpoints = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(checkpoints["model_state"])
        optimizer.load_state_dict(checkpoints["optimizer_state"])
        lr_scheduler.load_state_dict(checkpoints["lr_scheduler_state"])
        if ema_model:
            ema_model.load_state_dict(checkpoints["model_ema"])
    
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size if cfg.batch_size is not None else int(config["batch_size"]),
        sampler=torch.utils.data.RandomSampler(train_dataset),
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn)
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size if cfg.batch_size is not None else int(config["batch_size"]),
        sampler=torch.utils.data.SequentialSampler(valid_dataset),
        num_workers=2,
        pin_memory=True)
    
    
    for epoch in range(cfg.epochs): 
        train_epoch(config=config, epoch=epoch, model=model, 
                    ema_model=ema_model, device=device, data_loader=train_loader, 
                    optimizer=optimizer, criterion=train_criterion, limit=cfg.limit)

        lr_scheduler.step()
        if ema_model:
            valid_auc, valid_loss, _ = valid_test_epoch(
                                         name='Valid',
                                         epoch=epoch,
                                         model=ema_model,
                                         device=device,
                                         data_loader=valid_loader,
                                         criterion=valid_criterion,
                                         limit=cfg.limit)
        else:
            valid_auc, valid_loss, _ = valid_test_epoch(
                                         name='Valid',
                                         epoch=epoch,
                                         model=model,
                                         device=device,
                                         data_loader=valid_loader,
                                         criterion=valid_criterion,
                                         limit=cfg.limit)
                
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            ckpts = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "lr_scheduler_state": lr_scheduler.state_dict()}
            if ema_model:
                ckpts["ema_state"] = ema_model.state_dict()
            torch.save(ckpts, path)
            
        tune.report(Loss=valid_loss, AUC=valid_auc)
    print("Finished Training")
    
    
    
def main(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_dir = os.path.abspath(cfg.dataset_dir)
    
    config = {}
    hyperparam_mutations = {}
    
    ######### Recipe 1 #########
    ### lr optimization ###
    if cfg.lr_optim:
        config["lr"] = tune.qloguniform(1e-3, 1e-1, 1e-3)
        config["lr_warmup_decay"] = tune.qloguniform(1e-4, 1e-2, 1e-4)
        config["lr_warmup_epochs"] = tune.randint(3, 15)
        config["batch_size"] = tune.choice([32, 64, 128])
        
        hyperparam_mutations["lr"] = [1e-4, 1e-1]
        hyperparam_mutations["lr_warmup_decay"] = [1e-5, 1e-2]
        hyperparam_mutations["lr_warmup_epochs"] = [3, 15]
        hyperparam_mutations["batch_size"] = [32, 128]

    ######### Recipe 2 #########   
    if cfg.trivial:
        config["interpolation"] = tune.choice([InterpolationMode.BICUBIC, InterpolationMode.BILINEAR, InterpolationMode.NEAREST])
        config["num_magnitude_bins"] = tune.randint(5, 30)
        
        hyperparam_mutations["num_magnitude_bins"] = [5, 30]
    if cfg.random:
        config["interpolation"] = tune.choice([InterpolationMode.BICUBIC, InterpolationMode.BILINEAR, InterpolationMode.NEAREST])
        config["num_ops"] = tune.randint(2, 5)
        config["magnitude"] = tune.randint(5, 9)
        config["num_magnitude_bins"] = tune.randint(5, 30)
        
        hyperparam_mutations["num_ops"] = [2, 5]
        hyperparam_mutations["magnitude"] = [5, 9]
        hyperparam_mutations["num_magnitude_bins"] = [5, 30]

    ######### Recipe 4 #########  
    ### data aug ###
    if cfg.random_erase:
        config["random_erase_prob"] = [0.1, 0.2, 0.3]
        
        hyperparam_mutations["random_erase_prob"] = [0.1, 0.2, 0.3]

    ######### Recipe 5 ######### 
    ### label smoothing
    if cfg.label_smoothing:
        config["label_smoothing"] = tune.choice([0.05, 0.1, 0.15])
        
        hyperparam_mutations["label_smoothing"] = [0.05, 0.15]

    ######### Recipe 6 #########
    ### mixup ###
    if cfg.mixup:
        config["mixup_alpha"] = tune.quniform(0.1, 0.5, 0.1)
        
        hyperparam_mutations["mixup_alpha"] = [0.1, 0.5]
        
    ######### Recipe 7 #########
    ### cutmix ###
    if cfg.cutmix:
        config["cutmix_alpha"] = tune.quniform(0.4, 1.0, 0.1)  
        
        hyperparam_mutations["cutmix_alpha"] = [0.4, 1.0]

    ######### Recipe 8 #########
    ### weight decay tuning ###
    if cfg.wd_tune:
        config["weight_decay"] = tune.qloguniform(1e-5, 1e-3, 1e-6)
        
        hyperparam_mutations["weight_decay"] = [1e-5, 1e-3]

    ######### Recipe 10 #########
    ### model ema ###
    if cfg.ema:
        config["model_ema_steps"] = tune.qrandint(15, 50, 5)
        config["model_ema_decay"] = tune.uniform(0.99, 0.99998)
        
        hyperparam_mutations["model_ema_steps"] = [15, 50]
        hyperparam_mutations["model_ema_decay"] = [0.99, 0.99998]
        
    print(config)
    load_data(config, data_dir)
    
    if cfg.asha:
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric="AUC",
            mode="max",
        )  
    
    elif cfg.pbt:
        scheduler = scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="AUC",
            mode="max",
            hyperparam_mutations=hyperparam_mutations
        )
        
    elif cfg.pb2:
        scheduler = pb2.PB2(
            time_attr="training_iteration",
            metric="AUC",
            mode="max",
            hyperparam_bounds=hyperparam_mutations
        )
    else:
        raise Exception("The specified Scheduler is not considered in this work, Choose any of ['asha', 'ahb', 'pbt', 'pb2']")
    
    reporter = CLIReporter(
        metric_columns=["loss", "AUC", "training_iteration"])
        
    result = tune.run(
        partial(train, data_dir=data_dir),
        name=cfg.name,
        resources_per_trial={"cpu": cfg.cpus_per_trial, "gpu": cfg.gpus_per_trial},
        config=config,
        num_samples=cfg.num_samples,
        scheduler=scheduler,
        stop={"AUC": 0.99},
        resume="AUTO",
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("AUC", "max", "last")
    print(f"\nBest trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['Loss']:4.4f}")
    print(f"Best trial final validation AUC: {best_trial.last_result['AUC']:4.4f}")
    
    ################ Run inference with best model #####################
    _, _, test_dataset, num_classes = load_data(config, data_dir=data_dir)
    test_loader = torch.utils.data.DataLoader(
                                test_dataset,
                                batch_size=32,
                                sampler=torch.utils.data.SequentialSampler(test_dataset),
                                num_workers=2,
                                pin_memory=True)
    
    if "deit" in cfg.model:
        model = torch.hub.load("facebookresearch/deit:main", cfg.model+"_patch16_224", pretrained=True)
        out_chn = model.patch_embed.proj.out_channels
        model.patch_embed.proj = torch.nn.Conv2d(1, out_chn, kernel_size=(16, 16), stride=(16, 16))
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes)
    elif "resnet" in cfg.model:
        model = torchvision.models.__dict__[cfg.model](pretrained=True)
        out_chn = model.conv1.out_channels
        model.conv1 = torch.nn.Conv2d(1, out_chn, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes, bias=True)
    else:
        raise Exception("The specified model isn't considered for this work. Choose any ResNet or DEIT model.")
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    
    best_checkpoint_dir = best_trial.checkpoint.value
    ckpts = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    model.load_state_dict(ckpts["model_state"])
    
    
    criterion = BinaryCrossEntropy(smoothing=0.0).to(device)
    
    test_auc, test_loss, task_aucs = valid_test_epoch(name='Test', 
                                                      epoch=None, 
                                                      model=model, 
                                                      device=device, 
                                                      data_loader=test_loader, 
                                                      criterion=criterion,
                                                      limit=cfg.limit)

    print(f"\nAverage AUC over all pathologies: {test_auc:4.4f}")
    print(f"Test loss: {test_loss:4.4f}")                                 
    print(f"AUC for each task: {[round(x, 4) for x in task_aucs]}")
    
    return result



def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="Model Optimization Training For CXR ", add_help=add_help)
    parser.add_argument("--dataset_dir", default="../../../Music/work/data/", type=str, help="dataset path")
    parser.add_argument("--dataset", default="google", type=str, help="Choose one of ['nih', 'chex', 'mimic', 'pc', 'google', 'rsna', 'openi']")
    parser.add_argument("--seed", default=99, type=int, help="")
    parser.add_argument("--model", default="deit_tiny", type=str, help="Choose any ResNet or DEIT model, passing its name. Eg: resnet18, resnet50, deit_tiny, deit_base")
    parser.add_argument("--fext", action="store_true", default=False, help="Whether to perform feature extraction or finetune the entire model")
    parser.add_argument("--num_samples", default=5, type=int, help="Number of search trials")
    parser.add_argument("--epochs", default=5, type=int, help="")
    parser.add_argument("--cpus_per_trial", default=1, type=int, help="Number of CPUs for each trial")
    parser.add_argument("--gpus_per_trial", default=0, type=int, help="Number of GPUs for each tial. Automatic distributed training if multiple GPUs")
    parser.add_argument("--name", default="hparams_tune", type=str, help="Checkpoint directory name")
    parser.add_argument("--limit", default=None, type=int, help="Number of batches to use for training, validation and test. None to use entire data")
    ###### scheduler type
    parser.add_argument("--asha", action="store_true", default=False, help="Async Successive Halving")
    parser.add_argument("--pbt", action="store_true", default=False, help="Population Based Training")
    parser.add_argument("--pb2", action="store_true", default=False, help="Population Based Bandits")
    ###### pass to instantiate recipe                   
    parser.add_argument("--lr_optim", action="store_true", default=False, help="Pass this argument to tune lr, batch_size, and lr schedulers")
    parser.add_argument("--trivial", action="store_true", default=False, help="Pass this argument to use, and tune TrivialAugment data augmentation")
    parser.add_argument("--random", action="store_true", default=False, help="Pass this argument to use, and tune RandAugment data augmentation")
    parser.add_argument("--random_erase", action="store_true", default=False, help="Pass this argument to use, and tune Random Erasing")
    parser.add_argument("--label_smoothing", action="store_true", default=False, help="Pass this argument to use, and tune Label smoothing")
    parser.add_argument("--mixup", action="store_true", default=False, help="Pass this argument to use, and tune MixUp")
    parser.add_argument("--cutmix", action="store_true", default=False, help="Pass this argument to use, and tune CutMix")
    parser.add_argument("--wd_tune", action="store_true", default=False, help="Pass this argument to use, and tune Weight Decay")
    parser.add_argument("--ema", action="store_true", default=False, help="Pass this argument to use, and tune Model EMA")
    ###### pass to use optimized hparam                 
    parser.add_argument("--batch_size", default=None, type=int, help="")
    parser.add_argument("--lr", default=None, type=float, help="")
    parser.add_argument("--lr_warmup_epochs", default=None, type=int, help="")
    parser.add_argument("--lr_warmup_decay", default=None, type=float, help="")
    parser.add_argument("--weight_decay", default=None, type=float, help="")
    parser.add_argument("--smooth", default=None, type=float, help="")
    parser.add_argument("--mixup_alpha", default=None, type=float, help="")
    parser.add_argument("--cutmix_alpha", default=None, type=float, help="")
    parser.add_argument("--random_erase_prob", default=None, type=float, help="")
    parser.add_argument("--model_ema_steps", default=None, type=int, help="")
    parser.add_argument("--model_ema_decay", default=None, type=float, help="")
    parser.add_argument("--train_crop_size", default=None, type=int, help="")
    parser.add_argument("--interpolation", default=None, type=int, help="")
    parser.add_argument("--num_ops", default=None, type=int, help="")
    parser.add_argument("--magnitude", default=None, type=int, help="")
    parser.add_argument("--num_magnitude_bins", default=None, type=int, help="")
    # parser.add_argument("--", default=None, type=int, help="")
    
    return parser



if __name__ == "__main__":
    cfg = get_args_parser().parse_args()    
    ### seed for ray[tune] schedulers
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # You can change the number of GPUs per trial here:
    result = main(cfg)
