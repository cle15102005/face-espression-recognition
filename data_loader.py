import pandas as pd
def load_data(train_csv, val_csv, min_samples=2000, samples_per_class=500):
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