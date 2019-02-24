import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import warnings
from functools import partial
from pyll.base import TorchModel, AverageMeter
from pyll.session import PyLL
from pyll.utils.workspace import Workspace
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def main():
    session = PyLL()
    datasets = session.datasets
    model = session.model
    workspace = session.workspace
    config = session.config
    start_epoch = session.epoch
    best_f1 = 0
    
    if config.get_value("inference", False):
        dset = config.inference_params.dataset
        if dset in datasets:
            inference_loader = torch.utils.data.DataLoader(datasets[dset], batch_size=config.inference_params.batchsize,
                                                           num_workers=config.workers,
                                                           pin_memory=True, sampler=None, drop_last=False,
                                                           shuffle=False)
            out_dir = os.path.join(workspace.workspace_dir, "inference")
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            print("Inference path: {}".format(out_dir))
            inference(inference_loader, model, config, out_dir, dset)
            return
        else:
            print("Dataset '{}' not configured!".format(dset))
            exit(1)
    
    # data loader
    loader_train = torch.utils.data.DataLoader(
        datasets["train"],
        batch_size=config.training.batchsize, shuffle=True,
        num_workers=config.workers, pin_memory=False, sampler=None, drop_last=False)
    
    if "val" in datasets:
        loader_val = torch.utils.data.DataLoader(
            datasets["val"],
            batch_size=config.validation.batchsize, shuffle=False,
            num_workers=config.workers, pin_memory=False, drop_last=False)
        
        if config.evaluate:
            validate(loader_val, model, 0, None, None)
            return
        
        # Partial for validate function
        validate2 = partial(validate, loader=loader_val, model=model, config=config, workspace=workspace)
        # initial validation
        if start_epoch == 0 and config.initial_val:
            validate2(samples_seen=0)
    
    # LR scheduler
    # min mode: lr will be reduced when metric has stopped decreasing
    # max mode: lr will be reduced when metric has stopped increasing
    if config.get_value("schedule_lr", False):
        factor = config.lr_scheduler.get_value("factor", 0.1)
        patience = config.lr_scheduler.get_value("patience", 10)
        mode = config.lr_scheduler.get_value("mode", 'max')
        threshold = config.lr_scheduler.get_value("threshold", 1e-4)
        scheduler = ReduceLROnPlateau(session.optimizer, mode=mode, factor=factor, patience=patience,
                                      threshold=threshold)
    
    # Training Loop
    try:
        for epoch in range(start_epoch, config.training.epochs):
            # train for one epoch
            train(session, loader_train, epoch)
            
            if "val" in datasets:
                # evaluate on validation set
                val_f1 = validate2(samples_seen=(epoch + 1) * len(loader_train.dataset))
                
                # remember best performance and save checkpoint
                is_best = val_f1 > best_f1
                best_f1 = max(val_f1, best_f1)
                
                session.save_checkpoint(val_f1, is_best, filename="checkpoint-{}.pth.tar".format(epoch),
                                        model_best_filename='model_best_f1.pth.tar')
                
                write_checkpoint_log(session.workspace.checkpoint_dir, 'best_f1.txt',
                                     'Epoch: {}\nPerformance: {}'.format(epoch, best_f1))
                
                # update LR scheduler
                if config.get_value("schedule_lr", False):
                    scheduler.step(val_f1)
    finally:
        print("Done")


def train(session: PyLL, loader, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    # switch to train mode
    session.model.train()
    
    end = time.time()
    for i, batch in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = batch["input"]
        target = batch["target"]
        target = target.cuda(async=True)
        
        loss, output = session.train_step(input, target, epoch)
        
        if output.size(1) != loader.dataset.num_classes:
            output, _ = torch.split(output, loader.dataset.num_classes, dim=1)
        
        # record loss
        losses.update(loss.item(), input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % session.config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def validate(loader, model: TorchModel, config, samples_seen, workspace: Workspace):
    losses = AverageMeter()
    
    batch_size = loader.batch_size
    n_samples = len(loader.dataset)
    n_tasks = loader.dataset.num_classes
    predictions = np.zeros(shape=(n_samples, n_tasks))
    targets = np.zeros(shape=(n_samples, n_tasks), dtype=np.int)
    
    # switch to evaluate mode
    model.eval()
    
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        with torch.no_grad():
            input = batch["input"]
            target = batch["target"]
            target = target.cuda(async=True)
            
            if loader.dataset.patching:
                input = torch.squeeze(input)
            
            # compute output
            output = model(input)
            if loader.dataset.patching:
                loss = model.module.loss(output.mean(0).unsqueeze(0), target)
            else:
                loss = model.module.loss(output, target)
            
            if output.size(1) != n_tasks:
                output, _ = torch.split(output, n_tasks, dim=1)
            
            # store predictions and labels
            predictions[i * batch_size:(i + 1) * batch_size, :] = torch.sigmoid(output).mean(0).unsqueeze(
                0).cpu().data.numpy()
            targets[i * batch_size:(i + 1) * batch_size, :] = target.cpu().numpy().astype(np.int)
        
        # record loss
        losses.update(loss.item(), input.size(0))
    
    # store predictions
    if workspace is not None:
        np.savez_compressed(file="{}/step-{}-predictions.npz".format(workspace.results_dir, samples_seen),
                            X=predictions, Y=targets)
    
    # Metrics
    class_f1s = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_tasks):
            class_f1 = f1_score(y_true=targets[:, i], y_pred=(predictions[:, i] + 0.5).astype(int))
            class_f1s.append(class_f1)
    mean_f1 = float(np.mean(class_f1s))
    print(' * Validation F1-Score {f1:.3f}'.format(f1=mean_f1))
    return mean_f1


def inference(loader, model: TorchModel, config, out_dir, dset):
    batch_time = AverageMeter()
    
    batch_size = loader.batch_size
    n_samples = len(loader.dataset)
    n_tasks = loader.dataset.num_classes
    predictions = np.zeros(shape=(n_samples, n_tasks))
    sample_keys = []
    
    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    for i, batch in enumerate(loader):
        input = batch["input"]
        sample_keys.extend(batch["ID"])
        batch_avg = False
        
        if len(input.size()) == 5:
            input = input.view((-1,) + input.size()[-3:])
            batch_avg = True
        
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
        
        # compute output
        output = model(input_var)
        
        if output.size(1) != n_tasks:
            output, _ = torch.split(output, n_tasks, dim=1)
        
        if batch_avg:
            output = output.mean(0).unsqueeze(0)
        
        # store predictions and labels
        predictions[i * batch_size:(i + 1) * batch_size, :] = F.sigmoid(output).cpu().data.numpy()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % config.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(i, len(loader), batch_time=batch_time))
    
    # store predictions
    if out_dir is not None:
        with open(os.path.join(out_dir, "predictions_{}.csv".format(dset)), "w") as f:
            for i, sample in enumerate(sample_keys):
                preds = ','.join(map(str, predictions[i]))
                f.write("{},{}\n".format(sample, preds))
    
    print(" * Done!")


def write_checkpoint_log(folder, filename, text):
    with open(os.path.join(folder, filename), "w") as f:
        f.write(text)


if __name__ == '__main__':
    main()
