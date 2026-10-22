pip install pillow tqdm
import json
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

ROOT = Path("/content/Sentinel/datasets/crowdhuman")
IMAGE_DIR = ROOT / "images" / "Images"
TRAIN_ANN = ROOT / "annotation_train.odgt"
VAL_ANN = ROOT / "annotation_val.odgt"
OUTPUT = ROOT / "crowdhuman_yolo"

def ensure_dirs():
    folders = [OUTPUT / "images/train",OUTPUT / "images/val",OUTPUT / "labels/train",OUTPUT / "labels/val",]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def load_odgt(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def convert_bbox(box, img_w, img_h):
    x, y, w, h = box
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return (x_center,y_center,w,h)

def process_split(records, split):
    image_out = OUTPUT / f"images/{split}"
    label_out = OUTPUT / f"labels/{split}"
    processed = 0
    skipped = 0
    for record in tqdm(records):
        image_name = record["ID"] + ".jpg"
        image_path = IMAGE_DIR / image_name
        if not image_path.exists():
            skipped += 1
            continue
  
        img = Image.open(image_path)
        img_w, img_h = img.size
        shutil.copy(image_path, image_out / image_name)
        label_file = label_out / (record["ID"] + ".txt")
        with open(label_file, "w") as f:
            for obj in record["gtboxes"]:
                if obj["tag"] != "person":
                    continue
                if obj["head_attr"]["ignore"] == 1:
                    continue
                bbox = convert_bbox(obj["fbox"],img_w,img_h)
                line = "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*bbox)
                f.write(line)
        processed += 1
    print(f"{split} processed:", processed)
    print(f"{split} skipped:", skipped)

def create_dataset_yaml():
    with open(OUTPUT / "dataset.yaml", "w") as f:
        f.write(yaml)

def main():
    ensure_dirs()
    train = load_odgt(TRAIN_ANN)
    val = load_odgt(VAL_ANN)
    print("Train images:", len(train))
    print("Validation images:", len(val))

    process_split(train, "train")
    process_split(val, "val")
    create_dataset_yaml()
    print()
    print("Conversion complete.")

if __name__ == "__main__":
    main()
