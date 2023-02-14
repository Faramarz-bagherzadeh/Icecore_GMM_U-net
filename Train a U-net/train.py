import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model_V1 import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
#Hyper parameters
# Hyperparameters etc.


def train_fn (loader , model , optimizer, loss_fn , scaler):
    loop = tqdm (loader)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    for batch_idx , (data, targets) in enumerate (loop):
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(device = DEVICE)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            #predictions = torch.sigmoid(model(data))
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss = loss.item())



def main():
    LEARNING_RATE = 0.0001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1
    NUM_EPOCHS = 30
    NUM_WORKERS = 1
    IMAGE_HEIGHT = 2048  # 2024 originally
    IMAGE_WIDTH = 2048  # 2024 originally
    PIN_MEMORY = True
    LOAD_MODEL = False
    TRAIN_IMG_DIR = "D://Faramarz_data//mixed//train//input//"
    TRAIN_MASK_DIR = "D://Faramarz_data//mixed//train//output//"
    VAL_IMG_DIR = "D://Faramarz_data//mixed//test//input//"
    VAL_MASK_DIR = "D://Faramarz_data//mixed//test//output//"
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
           # A.Rotate(limit=35, p=1.0),
           # A.HorizontalFlip(p=0.5),
           # A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
           # A.Normalize(
           #     mean=[0.0, 0.0, 0.0],
           #    std=[1.0, 1.0, 1.0],
           #    max_pixel_value=255.0,
           # ),
            ToTensorV2(),
        ],
    )

    model = UNET().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    #loss_fn = nn.L1Loss()
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
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    #check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):


        if epoch <100:
            # check accuracy
            print ('validation accuracy')
            check_accuracy(val_loader, model, device=DEVICE)
            print ('train accuracy')
            check_accuracy(train_loader, model, device=DEVICE)
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),}
            save_checkpoint(checkpoint)

        train_fn(train_loader, model, optimizer, loss_fn, scaler)




        # print some examples to a folder
        #save_predictions_as_imgs(
        #    val_loader, model, folder="saved_images/", device=DEVICE)

if __name__ == '__main__':
    main()