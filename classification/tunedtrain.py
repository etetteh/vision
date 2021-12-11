from __future__ import print_function, division

import os
import copy
import time
import random
import argparse
import numpy as np
from urllib import request
from zipfile import ZipFile
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

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, pb2



def load_data(config, data_dir='./hymenoptera_data'):
    augment = [transforms.RandomResizedCrop(size=224, interpolation=InterpolationMode.BILINEAR),
               transforms.RandomHorizontalFlip()]
    ######### Recipe 2 #########           
    if cfg.random:
        augment.append(autoaugment.RandAugment(num_ops=config["num_ops"], magnitude=config["magnitude"], num_magnitude_bins=config["num_magnitude_bins"], interpolation=config["interpolation"]))
    elif cfg.num_ops:
        augment.append(autoaugment.RandAugment(num_ops=cfg.num_ops, magnitude=cfg.magnitude, num_magnitude_bins=cfg.num_magnitude_bins, interpolation=cfg.interpolation))
        
    if cfg.trivial:
        augment.append(autoaugment.TrivialAugmentWide(num_magnitude_bins=config["num_magnitude_bins"], interpolation=config["interpolation"]))
    elif cfg.num_magnitude_bins:
        augment.append(autoaugment.TrivialAugmentWide(num_magnitude_bins=cfg.num_magnitude_bins, interpolation=cfg.interpolation))

    augment.extend([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])  
    ######### Recipe 4 #########
    if cfg.random_erase:
        augment.append(transforms.RandomErasing(p=config["random_erase_prob"]))
    elif cfg.random_erase_prob:
        augment.append(transforms.RandomErasing(p=cfg.random_erase_prob))
    
    ######### Recipe 9 #########
    if cfg.fixres:
        augment[0] = transforms.RandomResizedCrop(size=config["train_crop_size"], interpolation=cfg.interpolate)  
    if cfg.train_crop_size:
        augment[0] = transforms.RandomResizedCrop(size=cfg.train_crop_size, interpolation=cfg.interpolate)  

    augment = transforms.Compose(augment)
    
    valid_augment = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
                       
    trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=augment)
    testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=valid_augment)
    num_classes = len(trainset.classes)
    return trainset, testset, num_classes
    
 
 
def train(config, checkpoint_dir=None, data_dir=None):
    torch.random.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    trainset, testset, num_classes = load_data(config, data_dir=data_dir)
    
    model = torch.hub.load("facebookresearch/deit:main", "deit_"+cfg.model+"_patch16_224", pretrained=False)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, num_classes)
    
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
        if cfg.bce:
            train_criterion = BinaryCrossEntropy(smoothing=0.0)
        else:
            train_criterion = SoftTargetCrossEntropy()
    elif cfg.label_smoothing:
        if cfg.bce:
            if cfg.smooth:
                train_criterion = BinaryCrossEntropy(smoothing=cfg.smooth)
            else:
                train_criterion = BinaryCrossEntropy(smoothing=config["label_smoothing"])
        else:
            if cfg.smooth:
                train_criterion = LabelSmoothingCrossEntropy(smoothing=cfg.smooth)
            else:
                train_criterion = LabelSmoothingCrossEntropy(smoothing=config["label_smoothing"])
    else:
        train_criterion = nn.CrossEntropyLoss()
    train_criterion = train_criterion.to(device)

    if cfg.bce:
        valid_criterion = torch.nn.BCEWithLogitsLoss()
    else:
        valid_criterion = torch.nn.CrossEntropyLoss()
    valid_criterion = valid_criterion.to(device)
    
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
    
    test_abs = int(len(trainset) * 0.8)
    train_subset, valid_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])
    
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=cfg.batch_size if cfg.batch_size is not None else int(config["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn)
    
    valid_loader = torch.utils.data.DataLoader(
        valid_subset,
        batch_size=cfg.batch_size if cfg.batch_size is not None else int(config["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True)
    
    
    for epoch in range(cfg.epochs): 
        train_one_epoch(epoch, train_loader, model, ema_model, optimizer, train_criterion, device=device)

        lr_scheduler.step()
        if ema_model:
            valid_loss, accuracy = validate(valid_loader, ema_model, valid_criterion, device=device)
        else:
            valid_loss, accuracy = validate(valid_loader, model, valid_criterion, device=device)
                
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            ckpts = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "lr_scheduler_state": lr_scheduler.state_dict()}
            if ema_model:
                ckpts["ema_state"] = ema_model.state_dict()
            torch.save(ckpts, path)
            
        tune.report(loss=valid_loss, accuracy=accuracy)
    print("Finished Training")
    
    
    
def train_one_epoch(epoch, train_loader, model, ema_model, optimizer, criterion, device="cpu"):
    running_loss = 0.0
    epoch_steps = 0
    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_steps += 1
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps))
            running_loss = 0.0

        if ema_model and i % cfg.model_ema_steps == 0:
            ema_model.update_parameters(model)
            if epoch < cfg.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                ema_model.n_averaged.fill_(0)


def validate(valid_loader, model, criterion, device="cpu"):
    valid_loss = 0.0
    valid_steps = 0
    total = 0
    correct = 0
    model.eval()
    for i, data in enumerate(valid_loader, 0):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            valid_loss += loss.cpu().numpy()
            valid_steps += 1
    
    valid_loss = (valid_loss / valid_steps)
    accuracy = correct / total
    return  valid_loss, accuracy


def test_accuracy(config, data_dir, model, device="cpu"):
    trainset, testset, num_classes = load_data(config, data_dir=data_dir)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
    
    
    
def main(cfg):
    torch.random.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_dir = os.path.abspath(cfg.data_dir)
    
    config = {}
    
    ######### Recipe 1 #########
    ### lr optimization ###
    if cfg.lr_optim:
        config["lr"] = tune.qloguniform(1e-4, 1e-1, 1e-5)
        config["lr_warmup_decay"] = tune.qloguniform(1e-5, 1e-3, 1e-6)
        config["lr_warmup_epochs"] = tune.qrandint(3, 15, 3)
        config["batch_size"] = tune.choice([16, 32, 64, 128])

    ######### Recipe 2 #########   
    config["interpolation"] = tune.choice([InterpolationMode.BICUBIC, InterpolationMode.BILINEAR, InterpolationMode.NEAREST])
    if cfg.trivial:
        config["num_magnitude_bins"] = tune.randint(15, 35)
    if cfg.random:
        config["num_ops"] = tune.randint(2, 5)
        config["magnitude"] = tune.randint(5, 9)
        config["num_magnitude_bins"] = tune.randint(15, 35)

    ######### Recipe 4 #########  
    ### data aug ###
    if cfg.random_erase:
        config["random_erase_prob"] = tune.quniform(0.1, 0.3, 0.05)

    ######### Recipe 5 ######### 
    ### label smoothing
    if cfg.label_smoothing:
        config["label_smoothing"] = tune.choice([0.05, 0.1, 0.15])

    ######### Recipe 6 #########
    ### mixup ###
    if cfg.mixup:
        config["mixup_alpha"] = tune.quniform(0.1, 0.5, 0.1)

    ######### Recipe 7 #########
    ### cutmix ###
    if cfg.cutmix:
        config["cutmix_alpha"] = tune.quniform(0.4, 1.0, 0.1)      

    ######### Recipe 8 #########
    ### weight decay tuning ###
    if cfg.wd_tune:
        config["weight_decay"] = tune.qloguniform(1e-5, 1e-3, 1e-6)

    ######### Recipe 9 #########   
    ### fixres ###
    if cfg.fixres:
        config["train_crop_size"] = tune.grid_search([176, 192, 208])    

    ######### Recipe 10 #########
    ### model ema ###
    if cfg.ema:
        config["model_ema_steps"] = tune.qrandint(15, 50, 5)
        config["model_ema_decay"] = tune.uniform(0.99, 0.99998)
    
    print(config)
    load_data(config, data_dir)
    
    if cfg.asha:
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric="accuracy",
            mode="max",
            max_t=cfg.epochs,
            grace_period=1,
            reduction_factor=2
        )
    
    if cfg.pbt:
        scheduler = scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="accuracy",
            mode="max",
            perturbation_interval=300.0,
            hyperparam_mutations=config
        )
        
    if cfg.pb2:
        scheduler = pb2.PB2(
            time_attr="training_iteration",
            metric="accuracy",
            mode="max",
            perturbation_interval=300.0,
            hyperparam_bounds=config
        )
    
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
        
    result = tune.run(
        partial(train, data_dir=data_dir),
        name=cfg.name,
        resources_per_trial={"cpu": cfg.cpus_per_trial, "gpu": cfg.gpus_per_trial},
        config=config,
        num_samples=cfg.num_samples,
        scheduler=scheduler,
        stop={"accuracy": 0.99},
        resume="AUTO",
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    
    
    model = torch.hub.load("facebookresearch/deit:main", "deit_"+cfg.model+"_patch16_224", pretrained=False)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, num_classes)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    checkpoints = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    model.load_state_dict(checkpoints["model_state"])

    test_acc = test_accuracy(config, data_dir, model, device)
    print("Best trial test set accuracy: {}".format(test_acc))
    
    return result    
    
    

def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="Model Optimization Training", add_help=add_help)
    parser.add_argument("--data_dir", default="./hymenoptera_data", type=str, help="dataset path")
    parser.add_argument("--seed", default=99, type=int, help="")
    parser.add_argument("--model", default="tiny", type=str, help="")
    parser.add_argument("--num_samples", default=5, type=int, help="")
    parser.add_argument("--epochs", default=5, type=int, help="")
    parser.add_argument("--cpus_per_trial", default=2, type=int, help="")
    parser.add_argument("--gpus_per_trial", default=0, type=int, help="")
    parser.add_argument("--name", default="hparams_tune", type=str, help="")
    ###### scheduler type
    parser.add_argument("--asha", action="store_true", default=False, help="")
    parser.add_argument("--pbt", action="store_true", default=False, help="")
    parser.add_argument("--pb2", action="store_true", default=False, help="")
    ###### pass to instantiate recipe                   
    parser.add_argument("--lr_optim", action="store_true", default=False, help="")
    parser.add_argument("--trivial", action="store_true", default=False, help="")
    parser.add_argument("--random", action="store_true", default=False, help="")
    parser.add_argument("--random_erase", action="store_true", default=False, help="")
    parser.add_argument("--label_smoothing", action="store_true", default=False, help="")
    parser.add_argument("--mixup", action="store_true", default=False, help="")
    parser.add_argument("--cutmix", action="store_true", default=False, help="")
    parser.add_argument("--wd_tune", action="store_true", default=False, help="")
    parser.add_argument("--fixres", action="store_true", default=False, help="")
    parser.add_argument("--ema", action="store_true", default=False, help="")
    parser.add_argument("--bce", action="store_true", default=False, help="")
    ###### pass to use optimized hparam                   
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
    
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # You can change the number of GPUs per trial here:
    result = main(cfg)