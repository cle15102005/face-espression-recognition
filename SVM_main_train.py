import pandas as pd
from SVM_src import SVMEmotionModel

LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def load_data(train_csv, val_csv):
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    X_train = df_train.drop(['Image', 'label'], axis=1).values
    y_train = df_train['label'].values

    X_val = df_val.drop(['Image', 'label'], axis=1).values
    y_val = df_val['label'].values

    return X_train, y_train, X_val, y_val


def main():
    train_csv = r'D:\PROJECT\virtual_env\Project ML\emotional_train_limited.csv'
    val_csv = r'D:\PROJECT\virtual_env\Project ML\emotional_val_limited.csv'

    print("🔄 Đang tải dữ liệu...")
    X_train, y_train, X_val, y_val = load_data(train_csv, val_csv)

    model = SVMEmotionModel(n_components=100, kernel='rbf', C=10, gamma='scale')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    model.evaluate(y_val, y_pred, labels=LABELS)


if __name__ == '__main__':
    main()
