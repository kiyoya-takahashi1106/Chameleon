import os
import pathlib
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET_ID = "gianmarco96/upmcfood101"

def get_project_root():
    try:
        return pathlib.Path(__file__).resolve().parent.parent
    except NameError:
        return pathlib.Path.cwd().resolve().parent

def load_env_from_file(env_path: pathlib.Path):
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        # 余計なクォートを除去
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        os.environ.setdefault(k, v)

def main():
    project_root = get_project_root()
    root = (project_root / "data" / "upmc_food101").resolve()
    root.mkdir(parents=True, exist_ok=True)

    # .env をプロジェクトルートから読み込み（KAGGLE_USERNAME / KAGGLE_KEY を期待）
    load_env_from_file(project_root / ".env")

    print(f"[*] project_root = {project_root}")
    print(f"[*] dataset root = {root}")
    print(f"[*] Using KAGGLE_USERNAME={os.environ.get('KAGGLE_USERNAME')!r}")

    # Kaggle からダウンロード＆解凍
    api = KaggleApi()
    api.authenticate()  # 環境変数の KAGGLE_USERNAME / KAGGLE_KEY を利用
    print(f"[*] Downloading {DATASET_ID} ...")
    api.dataset_download_files(DATASET_ID, path=str(root), unzip=True)

    print("[✓] Done.")
    print(f"Files are under: {root}")

if __name__ == "__main__":
    main()
