import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from CNN_src import CNNEmotionModel
from CNN_data_loader import load_data_and_generators

def main():
    # Tham số
    IMG_HEIGHT, IMG_WIDTH = 48, 48
    BATCH_SIZE = 64
    NUM_CLASSES = 6
    TRAIN_DIR = r'D:\PROJECT\virtual_env\Project ML\images\train'
    VAL_DIR = r'D:\PROJECT\virtual_env\Project ML\images\validation'

    # Load dữ liệu
    train_gen, val_gen = load_data_and_generators(TRAIN_DIR, VAL_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

    # Tính class weights để cân bằng
    classes = np.unique(train_gen.classes)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_gen.classes)
    class_weight_dict = dict(zip(classes, class_weights))

    # Khởi tạo và huấn luyện model
    model = CNNEmotionModel(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), num_classes=NUM_CLASSES)
    history = model.train(train_gen, val_gen, class_weights=class_weight_dict)

    # Vẽ loss và accuracy
    model.plot_history(history)

    # Đánh giá
    model.evaluate(val_gen)

if __name__ == '__main__':
    main()
