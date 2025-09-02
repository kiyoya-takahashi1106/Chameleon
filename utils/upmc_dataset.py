from pathlib import Path
import glob
from typing import List, Tuple, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class UPMCFood101Dataset(Dataset):
    """UPMC Food-101 dataset loader.

    - root: dataset root path (例: "data/UPMC_Food101")
    - split: "train" or "test"
    - img_size: 出力画像サイズ（正方）224 等
    - text_mode: "texts_txt"
    """
    def __init__(
        self,
        root: str,
        split: str,
        img_size: int = 224,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / "images" / split
        self.text_dir = self.root / "texts_txt"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        # クラス一覧（フォルダ名ソート）
        classes = sorted([p.name for p in self.img_dir.iterdir() if p.is_dir()])
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 画像パスと対応テキストパス（無ければ None）を収集
        self.samples: List[Tuple[Path, int, Optional[Path]]] = []
        for cls in self.classes:
            idx = self.class_to_idx[cls]
            pattern = str(self.img_dir / cls / "*")
            for img_path in sorted(glob.glob(pattern)):
                imgp = Path(img_path)
                base = imgp.stem
                text_path = None

                candidate = self.text_dir / cls / (base + ".txt")
                if candidate.exists():
                    text_path = candidate
                    textp = Path(text_path)

                self.samples.append((imgp, idx, textp))

        # transforms: RGB にしてリサイズ、ToTensor -> [0,1]
        self.tf = transforms.Compose([
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),  # -> (3, H, W), float32, [0,1]
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        imgp, label, textp = self.samples[i]
        # 画像読み込み
        with Image.open(imgp) as im:
            img = self.tf(im)

        # テキスト読み込み（無ければ空文字）
        text = ""
        if textp is not None and textp.exists():
            try:
                text = textp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
        return img, text, label
