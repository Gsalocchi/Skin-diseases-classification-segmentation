# data_processing.py
import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

load_dotenv()
TASK_3_TRAIN_LABELS_DIR = os.getenv("TASK_3_TRAIN_LABELS_DIR")
TASK_3_TRAIN_IMAGES_DIR = os.getenv("TASK_3_TRAIN_IMAGES_DIR")

TASK_3_VALIDATION_LABELS_DIR = os.getenv("TASK_3_VALIDATION_LABELS_DIR")
TASK_3_VALIDATION_IMAGES_DIR = os.getenv("TASK_3_VALIDATION_IMAGES_DIR")
TASK_3_TEST_LABELS_DIR = os.getenv("TASK_3_TEST_LABELS_DIR")
TASK_3_TEST_IMAGES_DIR = os.getenv("TASK_3_TEST_IMAGES_DIR")

print(f"Using CSV_PATH: {TASK_3_TRAIN_LABELS_DIR}")
print(f"Using IMAGES_DIR: {TASK_3_TRAIN_IMAGES_DIR}")
print(f"Using VAL CSV: {TASK_3_VALIDATION_LABELS_DIR}")
print(f"Using VAL IMAGES_DIR: {TASK_3_VALIDATION_IMAGES_DIR}")
print(f"Using TEST CSV: {TASK_3_TEST_LABELS_DIR}")
print(f"Using TEST IMAGES_DIR: {TASK_3_TEST_IMAGES_DIR}")

VAL_SIZE = 0.15
RANDOM_STATE = 42
BATCH_SIZE = 16
NUM_WORKERS = 1

class_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def build_dfs(
    csv_path: str = TASK_3_TRAIN_LABELS_DIR,
    images_dir: str = TASK_3_TRAIN_IMAGES_DIR,
    val_csv_path: str | None = None,
    val_images_dir: str | None = None,
    test_csv_path: str | None = None,
    test_images_dir: str | None = None,
):
    """Build train/val (and optional test) dataframes."""
    train_df = pd.read_csv(csv_path)
    train_df["label"] = train_df[class_cols].values.argmax(axis=1)
    train_df["path"] = train_df.apply(
        lambda row: os.path.join(images_dir, row["image"] + ".jpg"), axis=1
    )

    missing_train = train_df[~train_df["path"].apply(os.path.exists)]
    if not missing_train.empty:
        print("Warning: some train image files were not found, first few:")
        print(missing_train[["image", "path"]].head())

    val_df = None
    if val_csv_path:
        val_df = pd.read_csv(val_csv_path)
        val_df["label"] = val_df[class_cols].values.argmax(axis=1)
        val_images_dir = val_images_dir or TASK_3_VALIDATION_IMAGES_DIR
        val_df["path"] = val_df.apply(
            lambda row: os.path.join(val_images_dir, row["image"] + ".jpg"), axis=1
        )

        missing_val = val_df[~val_df["path"].apply(os.path.exists)]
        if not missing_val.empty:
            print("Warning: some val image files were not found, first few:")
            print(missing_val[["image", "path"]].head())
    else:
        train_df, val_df = train_test_split(
            train_df,
            test_size=VAL_SIZE,
            random_state=RANDOM_STATE,
            stratify=train_df["label"],
        )

    if test_csv_path:
        test_df = pd.read_csv(test_csv_path)
        test_df["label"] = test_df[class_cols].values.argmax(axis=1)
        test_images_dir = test_images_dir or TASK_3_TEST_IMAGES_DIR
        test_df["path"] = test_df.apply(
            lambda row: os.path.join(test_images_dir, row["image"] + ".jpg"), axis=1
        )

        missing_test = test_df[~test_df["path"].apply(os.path.exists)]
        if not missing_test.empty:
            print("Warning: some test image files were not found, first few:")
            print(missing_test[["image", "path"]].head())

        return train_df, val_df, test_df

    return train_df, val_df


# ----------------- TRANSFORMS ----------------- #

def get_transforms(image_size=(384, 384)):
    """
    Train:
      - random spatial + color augmentations
    Val/Test:
      - deterministic resize + center crop + normalize
    """

    train_tf = transforms.Compose([ # data augmentation resize 
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ),
        transforms.RandomRotation( # rotate between -180 and 180 degrees
            degrees=180,
            fill=0,
        ),
        transforms.RandomHorizontalFlip(p=0.5), #mirror horizontally and flip vertically
        transforms.RandomVerticalFlip(p=0.5), 
        transforms.ColorJitter( # change brightness, contrast, saturation, hue
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.02,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
            p=0.2,
        ),
        transforms.ToTensor(), # convert PIL image to tensor
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        transforms.RandomErasing(
            p=0.2,
            scale=(0.02, 0.06),
            ratio=(0.3, 3.3),
            value="random",
            inplace=False,
        ),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(image_size[0] * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    return train_tf, val_tf



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


def get_loaders(
    csv_path: str = TASK_3_TRAIN_LABELS_DIR,
    images_dir: str = TASK_3_TRAIN_IMAGES_DIR,
    val_csv_path: str = TASK_3_VALIDATION_LABELS_DIR,
    val_images_dir: str = TASK_3_VALIDATION_IMAGES_DIR,
    test_csv_path: str = TASK_3_TEST_LABELS_DIR,
    test_images_dir: str = TASK_3_TEST_IMAGES_DIR,
    image_size=(384, 384),
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    dfs = build_dfs(
        csv_path, images_dir, val_csv_path, val_images_dir, test_csv_path, test_images_dir
    )
    if len(dfs) == 3:
        train_df, val_df, test_df = dfs
    else:
        train_df, val_df = dfs

    train_tf, val_tf = get_transforms(image_size)

    train_dataset = SkinDataset(train_df, transform=train_tf)
    val_dataset   = SkinDataset(val_df,  transform=val_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,   # leave False for MPS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    if "test_df" in locals():
        test_dataset = SkinDataset(test_df, transform=val_tf)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
        return train_loader, val_loader, test_loader, train_df, val_df, test_df

    return train_loader, val_loader, train_df, val_df
