from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hog_features import extract_hog_features
from utils import DATA_DIR

TITANIC_TARGET_CANDIDATES = ("Survived", "2urvived")


def _build_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataframe(path):
    return pd.read_csv(Path(path))


def detect_target_column(columns, candidates=TITANIC_TARGET_CANDIDATES):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Unable to find a label column. Tried: {list(candidates)}")


def _first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the expected files exist: {paths}")


def get_titanic_paths():
    titanic_dir = DATA_DIR / "titanic"
    train_path = _first_existing(
        [
            titanic_dir / "train.csv",
            titanic_dir / "titanic_train.csv",
            titanic_dir / "titanic_train_knn.csv",
        ]
    )
    test_path = _first_existing(
        [
            titanic_dir / "test.csv",
            titanic_dir / "titanic_test.csv",
            titanic_dir / "titanic_test_knn.csv",
        ]
    )
    return train_path, test_path


def preprocess_titanic_dataframe(df, fit=True, preprocess_info=None):
    df = df.copy()
    target_col = detect_target_column(df.columns)

    zero_columns = [
        col
        for col in df.columns
        if col != target_col and pd.api.types.is_numeric_dtype(df[col]) and df[col].fillna(0).eq(0).all()
    ]
    drop_columns = zero_columns + [col for col in ("Passengerid", "PassengerId") if col in df.columns]
    df = df.drop(columns=drop_columns, errors="ignore")

    y = df[target_col].astype(int).to_numpy() if target_col in df.columns else None
    X = df.drop(columns=[target_col], errors="ignore")

    categorical_columns = [col for col in ("Sex", "Embarked", "Pclass") if col in X.columns]
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    if fit:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", _build_one_hot_encoder()),
            ]
        )

        transformer = ColumnTransformer(
            [
                ("numeric", numeric_pipeline, numeric_columns),
                ("categorical", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
        )
        X_processed = transformer.fit_transform(X)
        feature_names = transformer.get_feature_names_out().tolist()
        preprocess_info = {
            "transformer": transformer,
            "target_col": target_col,
            "feature_names": feature_names,
            "drop_columns": drop_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
        }
        return X_processed, y, preprocess_info

    transformer = preprocess_info["transformer"]
    X_processed = transformer.transform(X)
    return X_processed, y


def load_titanic_train_test():
    train_path, test_path = get_titanic_paths()
    train_df = load_dataframe(train_path)
    test_df = load_dataframe(test_path)
    X_train, y_train, preprocess_info = preprocess_titanic_dataframe(train_df, fit=True)
    X_test, y_test = preprocess_titanic_dataframe(test_df, fit=False, preprocess_info=preprocess_info)
    return X_train, y_train, X_test, y_test, preprocess_info, train_df, test_df


def preprocess_house_dataframe(df, fit=True, preprocess_info=None):
    df = df.copy()
    if "y" not in df.columns:
        raise ValueError("Expected the house price target column to be named 'y'.")

    y = df["y"].to_numpy()
    X = df.drop(columns=["y"])
    feature_names = X.columns.tolist()

    if fit:
        transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        X_processed = transformer.fit_transform(X)
        preprocess_info = {
            "transformer": transformer,
            "feature_names": feature_names,
        }
        return X_processed, y, preprocess_info

    transformer = preprocess_info["transformer"]
    X_processed = transformer.transform(X)
    return X_processed, y


def load_house_splits(test_size=0.2, random_state=42):
    house_path = DATA_DIR / "house" / "house_data.csv"
    df = load_dataframe(house_path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    X_train, y_train, preprocess_info = preprocess_house_dataframe(train_df, fit=True)
    X_test, y_test = preprocess_house_dataframe(test_df, fit=False, preprocess_info=preprocess_info)
    return X_train, y_train, X_test, y_test, preprocess_info, train_df, test_df


def _collect_image_records(root_dir, selected_classes=None):
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(root_dir)

    available_classes = sorted(path.name for path in root_dir.iterdir() if path.is_dir())
    if selected_classes:
        requested = [str(label) for label in selected_classes]
        class_names = [label for label in requested if label in available_classes]
        if len(class_names) != len(requested):
            missing = sorted(set(requested) - set(class_names))
            raise ValueError(f"Missing requested classes under {root_dir}: {missing}")
    else:
        class_names = available_classes

    records = []
    for class_name in class_names:
        class_dir = root_dir / class_name
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                records.append((image_path, class_name))
    return class_names, records


def _sample_records(records, max_samples, seed):
    if max_samples is None or len(records) <= max_samples:
        return records

    indices = np.arange(len(records))
    labels = np.array([label for _, label in records])

    if max_samples < len(set(labels)):
        rng = np.random.default_rng(seed)
        sampled = rng.choice(indices, size=max_samples, replace=False)
        return [records[index] for index in sorted(sampled)]

    sampled, _ = train_test_split(
        indices,
        train_size=max_samples,
        random_state=seed,
        stratify=labels,
    )
    return [records[index] for index in sorted(sampled)]


def _load_single_image(image_path, image_size, color_mode):
    flag = cv2.IMREAD_GRAYSCALE if color_mode == "grayscale" else cv2.IMREAD_COLOR
    image = cv2.imread(str(image_path), flag)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    if color_mode == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_image_dataset(
    root_dir,
    image_size,
    color_mode,
    use_hog=False,
    selected_classes=None,
    max_samples=None,
    seed=42,
):
    class_names, records = _collect_image_records(root_dir, selected_classes=selected_classes)
    records = _sample_records(records, max_samples=max_samples, seed=seed)

    images = []
    labels = []
    label_to_index = {label: index for index, label in enumerate(class_names)}

    for image_path, class_name in records:
        image = _load_single_image(image_path, image_size=image_size, color_mode=color_mode)
        images.append(image)
        labels.append(label_to_index[class_name])

    images = np.asarray(images)
    labels = np.asarray(labels)

    if use_hog:
        features = extract_hog_features(images)
    else:
        features = images.astype(np.float32)
        if features.ndim == 4:
            features = features / 255.0
        else:
            features = features / 255.0
        features = features.reshape(features.shape[0], -1)

    return features, labels, class_names, images


def load_image_train_test(dataset_name, use_hog=False, selected_classes=None, max_train=None, max_test=None, seed=42):
    dataset_dir = DATA_DIR / dataset_name

    if dataset_name == "mnist":
        image_size = (28, 28)
        color_mode = "grayscale"
    elif dataset_name == "cifar10":
        image_size = (32, 32)
        color_mode = "rgb"
    else:
        raise NotImplementedError(f"Unsupported image dataset: {dataset_name}")

    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    X_train, y_train, class_names, train_images = load_image_dataset(
        train_dir,
        image_size=image_size,
        color_mode=color_mode,
        use_hog=use_hog,
        selected_classes=selected_classes,
        max_samples=max_train,
        seed=seed,
    )
    X_test, y_test, _, test_images = load_image_dataset(
        test_dir,
        image_size=image_size,
        color_mode=color_mode,
        use_hog=use_hog,
        selected_classes=selected_classes,
        max_samples=max_test,
        seed=seed + 1,
    )

    metadata = {
        "dataset_name": dataset_name,
        "class_names": class_names,
        "image_size": image_size,
        "color_mode": color_mode,
        "use_hog": use_hog,
        "selected_classes": selected_classes or class_names,
    }
    return X_train, y_train, X_test, y_test, metadata, train_images, test_images
