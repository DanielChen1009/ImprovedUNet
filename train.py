import gc

import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import ImprovedUNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
# Perhaps change learning rate...
LEARNING_RATE = 1e-4
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    DEVICE = torch.device("cpu")

else:
    DEVICE = torch.device("cpu")
torch.set_flush_denormal(True)
BATCH_SIZE = 1
CLASSES = 5
NUM_EPOCHS = 20
NUM_WORKERS = 2
IMAGE_HEIGHT = 512  # 3000 originally
IMAGE_WIDTH = 512  # 4000 originally
INTERPOLATION_TYPE = cv2.INTER_AREA
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/xview/train/images"
TRAIN_MASK_DIR = "data/xview/train/targets"
VAL_IMG_DIR = "data/xview/train/images"
VAL_MASK_DIR = "data/xview/train/targets"


def train_fn(loader, model, optimizer, loss_fn, scaler, loss_sum):
    loop = tqdm(loader)
    for batch_idx, (data_pre, data_post, targets) in enumerate(loop):
        data_pre = data_pre.to(device=DEVICE)
        data_post = data_post.to(device=DEVICE)

        targets = targets.to(device=DEVICE)

        # forward
        predictions = model(data_pre, data_post)
        loss = loss_fn(predictions, targets.long())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # update tqdm loop
        loss_sum += loss.item()
        loop.set_postfix(loss=(loss_sum / (batch_idx + 1)))
    print(loss_sum / 157)


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=INTERPOLATION_TYPE),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        additional_targets={
            'pre': 'image'
        }
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=INTERPOLATION_TYPE),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        additional_targets={
            'pre': 'image'
        }
    )

    model = ImprovedUNet(in_channels=3, out_channels=CLASSES).to(device=DEVICE)
    # focal_loss = torch.hub.load(
    #     'adeelh/pytorch-multi-class-focal-loss',
    #     model='FocalLoss',
    #     alpha=torch.tensor([1.0/13, 2.0/13, 2.0/13, 3.0/13, 5.0/13]),
    #     gamma=2,
    #     reduction='mean',
    #     force_reload=False
    # )
    # loss_fn = focal_loss.to(device=DEVICE)
    loss_fn = nn.CrossEntropyLoss().to(device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint_ImprovedUNet.pth.tar", map_location='cpu')["state_dict"], model)
        optimizer.load_state_dict(torch.load("checkpoint_ImprovedUNet.pth.tar", map_location='cpu')["optimizer"])
    model = model.to(device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    check_accuracy(val_loader, model, device=DEVICE)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, 0)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename='checkpoint_ImprovedUNet.pth.tar')

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)


if __name__ == "__main__":
    main()
