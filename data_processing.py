# data_processing.py
import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# load .env if you want to keep paths there
load_dotenv()

# read paths from env or hardcode them
CSV_PATH = os.getenv("TASK_3_TRAIN_LABELS_DIR")
IMAGES_DIR = os.getenv("TASK_3_TRAIN_IMAGES_DIR")
print(f"Using CSV_PATH: {CSV_PATH}")
print(f"Using IMAGES_DIR: {IMAGES_DIR}")

VAL_SIZE = 0.15
RANDOM_STATE = 42
BATCH_SIZE = 16
NUM_WORKERS = 1

# columns in the order you showed
class_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def build_dfs(csv_path: str = CSV_PATH, images_dir: str = IMAGES_DIR):
    df = pd.read_csv(csv_path)

    # turn one-hot into single label
    df["label"] = df[class_cols].values.argmax(axis=1)

    def build_path(row):
        return os.path.join(images_dir, row["image"] + ".jpg")

    df["path"] = df.apply(build_path, axis=1)

    # optional check
    missing = df[~df["path"].apply(os.path.exists)]
    if not missing.empty:
        print("Warning: some image files were not found, first few:")
        print(missing[["image", "path"]].head())

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    return train_df, val_df


class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = int(row["label"])

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_transforms(image_size=(384, 384)):
    train_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return train_tf, val_tf


def get_loaders(
    csv_path: str = CSV_PATH,
    images_dir: str = IMAGES_DIR,
    image_size=(384, 384),
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    train_df, val_df = build_dfs(csv_path, images_dir)
    train_tf, val_tf = get_transforms(image_size)

    train_dataset = SkinDataset(train_df, transform=train_tf)
    val_dataset = SkinDataset(val_df, transform=val_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader, train_df, val_df

