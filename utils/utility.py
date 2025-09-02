import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List

def set_seed(seed :int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



# loss1つとaccuracy1つをプロットする関数
def plot_training_progress(epochs: List[int], losses: List[float], accuracies: List[float], 
                           save_path: str = None, show_plot: bool = True):
    """
    トレーニングの進行状況をプロットする関数

    Args:
        epochs: エポック番号のリスト
        losses: 各エポックの平均lossのリスト
        accuracies: 各エポックの精度のリスト
        save_path: グラフを保存するパス（Noneの場合は保存しない）
        show_plot: グラフを表示するかどうか
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss のプロット
    ax1.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Progress')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy のプロット
    ax2.plot(epochs, accuracies, 'r-', linewidth=2, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Accuracy Progress')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()



# img_loss, text_loss, combined_loss, accuracyをプロットする関数
def plot_multi_loss_progress(
    epochs: List[int],
    img_losses: List[float],
    text_losses: List[float],
    losses: List[float],
    accuracies: List[float],
    save_path: str = None,
    show_plot: bool = True
):
    """
    画像loss、テキストloss、合計loss、accuracyを1枚のグラフにプロットする関数

    Args:
        epochs: エポック番号のリスト
        img_losses: 画像Lossのリスト
        text_losses: テキストLossのリスト
        losses: 学習Lossのリスト
        accuracies: 精度のリスト
        save_path: グラフを保存するパス（Noneの場合は保存しない）
        show_plot: グラフを表示するかどうか
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Loss plots
    ax1.plot(epochs, img_losses, label="Image Loss", color="blue", linewidth=2)
    ax1.plot(epochs, text_losses, label="Text Loss", color="green", linewidth=2)
    ax1.plot(epochs, losses, label="Training Loss", linestyle="--", color="black", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Progress")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Accuracy plot on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, accuracies, label="Test Accuracy", color="red", marker="o", linestyle=":", linewidth=2, markersize=4)
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    
    if save_path:
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()