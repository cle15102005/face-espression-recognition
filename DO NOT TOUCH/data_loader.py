import pandas as pd

import numpy as np
import pandas as pd
import os
from PIL import Image

#flatten images
def flatten_image(folder, data_dir, categories, label_map):
    X = []
    y = []
    for category in categories:
        label = label_map[category]
        path = os.path.join(data_dir, folder, category)
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = Image.open(img_path).convert('L').resize((48, 48))
            arr = np.array(img).reshape(-1) / 255.0  # chuẩn hóa luôn
            X.append(arr)
            y.append(label)
    return np.array(X), np.array(y)

#generate csv file from image file
def generate_csv():
    # Cấu hình thư mục
    data_dir = r"D:\images"
    categories = ["angry", "disgust", "fear", "happy","neutral", "sad", "surprise"]
    label_map = {cat: idx for idx, cat in enumerate(categories)}

    # Load và chuẩn hóa
    X_train, y_train = flatten_image("train", data_dir, categories, label_map)
    X_val, y_val = flatten_image("validation", data_dir, categories, label_map)

    # Ghép lại để lưu CSV
    train_df = pd.DataFrame(X_train)
    train_df['label'] = y_train

    val_df = pd.DataFrame(X_val)
    val_df['label'] = y_val

    # Xuất file CSV
    train_df.to_csv("train_face_emotion1.csv", index=False)
    val_df.to_csv("val_face_emotion1.csv", index=False)

    print("Xuất CSV thành công! File: train_face_emotion1.csv, val_face_emotion1.csv")

#load data from csv file
def load_data(train_csv, val_csv, min_samples=2000, samples_per_class=3000):
    df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Filter valid classes
    class_counts = df['label'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index.tolist()

    df = df[df['label'].isin(valid_classes)].reset_index(drop=True)
    val_df = val_df[val_df['label'].isin(valid_classes)].reset_index(drop=True)

    if samples_per_class:
        df = df.groupby("label").apply(lambda x: x.sample(min(samples_per_class, len(x)), random_state=42)).reset_index(drop=True)

    X_train_raw = df.drop(columns=["Image", "label"]).values
    y_train = df["label"].values

    X_val_raw = val_df.drop(columns=["Image", "label"]).values
    y_val = val_df["label"].values
    
    return X_train_raw, X_val_raw, y_train, y_val