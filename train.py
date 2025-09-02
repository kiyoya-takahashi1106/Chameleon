# データセットを用いて直接学習する.
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
import torch.optim as optim
from model.chameleon import Chameleon
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils.upmc_dataset import UPMCFood101Dataset
from utils.utility import set_seed, plot_multi_loss_progress

print(torch.__version__)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dataset_name", default="UPMC_Food101", type=str)
    parser.add_argument("--class_num", default=101, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Chameleon(img_size=args.img_size, class_num=args.class_num)

    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    # datasets
    train_dataset = UPMCFood101Dataset(root=f"data/{args.dataset_name}", split="train", img_size=args.img_size)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_dataset = UPMCFood101Dataset(root=f"data/{args.dataset_name}", split="test", img_size=args.img_size)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device_ids = [i for i in range(torch.cuda.device_count())]
    print(device_ids)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    
    # モデル全体をGPUに移動
    model = model.to(device)
    
    # Text2ImgVecのAnglEモデルも確実にGPUに移動
    if hasattr(model, 'module'):  # DataParallelの場合
        model.module.text2img.to(device)
    else:
        model.text2img.to(device)

    acc_list = []
    img_loss_list = []
    text_loss_list = []
    loss_list = []

    for epoch in tqdm(range(args.epochs)):
        model.train()
        img_avg_loss = []
        text_avg_loss = []
        avg_loss = []

        for _, batch in enumerate(tqdm(train_data_loader)):
            # バッチから画像、テキスト、ラベルを取得
            images, texts, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            # モデルの順伝搬
            img_logits, text_logits = model(images, texts)

            img_p = F.softmax(img_logits, dim=1)
            text_p = F.softmax(text_logits, dim=1)

            # 画像とテキストの各loss (学習には使わない)
            img_loss = F.cross_entropy(img_logits, labels)
            text_loss = F.cross_entropy(text_logits, labels)
            img_avg_loss.append(img_loss.item())
            text_avg_loss.append(text_loss.item())
            
            # 画像とテキストの予測を平均化
            combined_p = (img_p + text_p) / 2

            loss = F.nll_loss((combined_p + 1e-8).log(), labels)
            avg_loss.append(loss.item())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        img_avg_loss = np.mean(img_avg_loss)
        text_avg_loss = np.mean(text_avg_loss)
        epoch_avg_loss = np.mean(avg_loss)

        img_loss_list.append(img_avg_loss)
        text_loss_list.append(text_avg_loss)
        loss_list.append(epoch_avg_loss)
        print(f"Epoch {epoch}, loss: {epoch_avg_loss}, img_loss: {img_avg_loss}, text_loss: {text_avg_loss}")

        # Test
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for _, batch in enumerate(tqdm(test_data_loader)):
                images, texts, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                img_logits, text_logits = model(images, texts)
                combined_logits = (img_logits + text_logits) / 2

                predictions = torch.max(combined_logits, 1)[1]
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            print("correct, total:", correct, total)

        acc = correct / total
        acc_list.append(acc)
        print(f"Epoch {epoch} Acc: {acc}")
        
        # グラフ描画
        epochs_range = list(range(len(acc_list)))
        plot_multi_loss_progress(
            epochs=epochs_range, 
            img_losses=img_loss_list,
            text_losses=text_loss_list,
            losses=loss_list, 
            accuracies=acc_list,
            save_path="result/train.png",
            show_plot=False
        )
        
        if (acc >= max(acc_list)):
            torch.save(model.state_dict(),
                       f"saved_models/{args.dataset_name}_epoch{epoch}_{acc:.4f}_seed{args.seed}.pth")
                       
    print("best acc: ", max(acc_list))
    return


if __name__ == "__main__":
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    set_seed(_args.seed)
    train(_args)