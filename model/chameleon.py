import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from sentence_transformers import SentenceTransformer


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

# ImageNetのavg/varで正規化
def imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    """
    [0, 1] => ImageNetのavg/varで正規化 (B, 3, H, W)
    """
    if (x.dtype != torch.float32) and (x.dtype != torch.float64):
        x = x.float()
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std  = IMAGENET_STD.to(x.device, x.dtype)
    return (x - mean) / std


# テキストをangleでembeddingし、画像サイズにリサイズ
class Text2ImgVec(nn.Module):
    def __init__(self, dim: int, img_size: int, 
                 model_name: str = "WhereIsAI/UAE-Large-V1"):
        """
        args:
            dim: 出力ベクトル次元（1024）
            img_size: 出力画像のサイズ
        """
        super().__init__()
        self.sentence_model = SentenceTransformer(model_name)
        self.dim = dim
        self.img_size = img_size
        
        # 768次元から1024次元に変換
        # self.projection = nn.Linear(768, dim)

    def forward(self, x: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
    
        # 1. テキストをベクトル化 (B, 768)
        embeddings = self.sentence_model.encode(x, convert_to_tensor=True, device=device)
        # detach()とclone()の両方を使用して、新しいテンソルを作成
        embeddings = embeddings.detach().clone()

        # 2. embeddings を float Tensor として扱う
        x = embeddings.to(device=device, dtype=torch.float32)

        # 3. [0, 1]に正規化
        eps = 1e-6
        x_min = x.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        x = (x - x_min) / (x_max - x_min + eps)

        # 4. reshape (B, 1, √d, √d)
        d = self.dim
        s = int(math.ceil(math.sqrt(d)))
        img_vec = x.view(-1, 1, s, s)

        # 5. resize (B, 1, img_size, img_size)
        img_vec = F.interpolate(img_vec, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        # 6. channelを3に複製 (B, 3, img_size, img_size)
        img_vec = img_vec.repeat(1, 3, 1, 1)

        # 7. ImageNetのavg/varで正規化
        img_vec = imagenet_norm(img_vec)

        return img_vec


# main model
class Chameleon(nn.Module):
    def __init__(self, img_size: int, class_num: int,):
        """
        args:
            dim: 出力ベクトル次元 d
            img_size: 出力画像のサイズ
            vit_model: 使用するViTモデル名
        """
        super().__init__()
        self.dim = 1024
        self.text2img = Text2ImgVec(self.dim, img_size)
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # ViTの分類ヘッドをclass_num数に変更
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, class_num)

    def forward(self, img_x: torch.Tensor, text_x: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. テキストを画像ベクトルに変換 (B, 3, img_size, img_size)
        text_x = self.text2img(text_x)

        # 2. 画像とテキスト画像ベクトルを同じVITに通す
        img_x = imagenet_norm(img_x)
        img_x = self.vit(img_x)
        text_x = self.vit(text_x)

        return img_x, text_x