import pandas as pd
from keras import preprocessing
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

def limit_sample():
    # === Load file gốc ===
    train_df = pd.read_csv("train_face_emotion_limited.csv")
    val_df = pd.read_csv("val_face_emotion_limited.csv")

    # === Lấy mỗi lớp 3000 mẫu từ train ===
    train_filtered = (
        train_df.groupby('label', group_keys=False)
        .apply(lambda x: x.sample(n=3000, random_state=42))
    )

    # === Lấy mỗi lớp 900 mẫu từ validation ===
    val_filtered = (
        val_df.groupby('label', group_keys=False)
        .apply(lambda x: x.sample(n=900, random_state=42))
    )

    # === Gán chỉ số dòng (Image x) ===
    train_filtered.index = [f"Image {i+1}" for i in range(len(train_filtered))]
    val_filtered.index = [f"Image {i+1}" for i in range(len(val_filtered))]

    # === Xuất ra file mới ===
    train_filtered.to_csv("emotional_train_limited.csv", index=True, index_label="Image")
    val_filtered.to_csv("emotional_val_limited.csv", index=True, index_label="Image")

    print("✅ Đã xuất 2 file giới hạn mẫu theo mỗi lớp:")
    print("   ➤ emotional_train_limited.csv (3000/lớp)")
    print("   ➤ emotional_val_limited.csv (900/lớp)")

def image_augmentation():
    input_dir = r"D:\images\validation\disgust"# Thư mục chứa ảnh disgust gốc (48x48 hoặc lớn hơn)
    output_dir = "augmented_disgust_val"      # Thư mục xuất ảnh tăng cường
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo bộ biến đổi dữ liệu
    datagen = preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Số ảnh tăng cường muốn tạo mỗi ảnh gốc
    AUG_PER_IMAGE = 4

    count = 0
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(input_dir, fname)
            img = preprocessing.image.load_img(img_path, color_mode='grayscale', target_size=(48, 48))
            x = preprocessing.image.img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Tạo ảnh mới từ ảnh gốc
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir,
                                    save_prefix="aug", save_format='png'):
                i += 1
                count += 1
                if i >= AUG_PER_IMAGE:
                    break

    print(f"✅ Đã tạo {count} ảnh tăng cường từ thư mục {input_dir}")
    
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