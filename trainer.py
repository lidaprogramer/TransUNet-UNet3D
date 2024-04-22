
import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import cv2
from PIL import Image

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args['base_lr']
    num_classes = args['num_classes']
    batch_size = args['batch_size'] * args['n_gpu']
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args['root_path'], list_dir=args['list_dir'], split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args['img_size'], args['img_size']])]))
    print("The length of train set is: {}".format(len(db_train)))
    
    def worker_init_fn(worker_id):
        random.seed(args['seed'] + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args['n_gpu'] > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args['max_epochs']
    max_iterations = args['max_epochs'] * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 30  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 4) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

def validate_model(model, data_directory, batch_size, ce_loss, dice_loss, device, writer, iter_num):
    model.eval()
    val_loss = 0
    total_samples = 0
    with torch.no_grad():
        images_list = []
        labels_list = []
        for file in os.listdir(data_directory):
            full_path = os.path.join(data_directory, file)
            data = np.load(full_path)
            image, label = data['image'], data['label']

            # Convert numpy arrays to tensors and add batch dimension if missing
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
            label_tensor = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float().to(device)

            images_list.append(image_tensor)
            labels_list.append(label_tensor)
            
            kol = 0
            # Check if we have gathered a full batch
            if len(images_list) == batch_size:
                batch_images = torch.cat(images_list)
                batch_labels = torch.cat(labels_list)
                loss, outputs = process_batch(batch_images, batch_labels, model, ce_loss, dice_loss)
                images_list = []
                labels_list = []
                writer.add_scalar('info/val_loss', loss, iter_num+kol)
                kol += 1

                if kol % 10 == 0:
                    image = batch_images[1, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('val/Image', image, iter_num+kol)
                    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    writer.add_image('val/Prediction', outputs[1, ...] * 50, iter_num+kol)
                    labs = batch_labels[1, ...].unsqueeze(0) * 50
                    writer.add_image('val/GroundTruth', labs, iter_num+kol)    

        # Process the last batch if it has fewer than batch_size images
        if images_list:
            batch_images = torch.cat(images_list)
            batch_labels = torch.cat(labels_list)
            loss, outputs = process_batch(batch_images, batch_labels, model, ce_loss, dice_loss)
            val_loss += loss
            writer.add_scalar('info/val_loss', loss, iter_num+kol)


    model.train()  # Reset to training mode


def process_batch(images, labels, model, ce_loss, dice_loss):
    outputs = model(images)
    loss_ce = ce_loss(outputs, labels[:].squeeze().long())
    loss_dice = dice_loss(outputs, labels.squeeze(), softmax=True)
    loss = 0.5 * loss_ce + 0.5 * loss_dice
    return loss, outputs


def trainer_penguin(args, model, snapshot_path):
    
    from datasets.dataset_penguin import Penguin_dataset, RandomGenerator
    logging.basicConfig(filename="/home/ubuntu/files/project_TransUNet/model/vit_checkpoint/imagenet21k/log_penguin/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args['base_lr']
    num_classes = args['num_classes']
    batch_size = args['batch_size'] * args['n_gpu']
    # max_iterations = args.max_iterations
    db_train = Penguin_dataset(base_dir=args['root_path'], list_dir=args['list_dir'], split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args['img_size'], args['img_size']])]))
    print("The length of train set is: {}".format(len(db_train)))
    
    
    def worker_init_fn(worker_id):
        random.seed(args['seed'] + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args['n_gpu'] > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter("/home/ubuntu/files/project_TransUNet/model/vit_checkpoint/imagenet21k/log_penguin/")
    iter_num = 0
    max_epoch = args['max_epochs']
    max_iterations = args['max_epochs'] * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        
        validate_model(model, '/home/ubuntu/files/project_TransUNet/data/Penguin/val_224', 24, ce_loss, dice_loss, 'cuda', writer, iter_num)
          
        save_interval = 3  # int(max_epoch/6)
        if epoch_num  % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break       

    writer.close()
    return "Training Finished!"