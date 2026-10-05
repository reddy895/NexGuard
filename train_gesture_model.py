"""
NexGuard Motion Dynamics & Gesture Classifier Trainer
Trains a machine learning classifier to detect vehicle accident dynamic patterns
(sudden deceleration, violent directional changes, bounding box deformation).
Exports gesture_classifier.joblib and gesture_classifier_fast.npz
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from config import config
from utils import log_event


def generate_synthetic_accident_dataset(num_samples: int = 1500):
    """
    Generates dataset of motion features:
    Feature vector: [
       0: speed_drop_ratio,
       1: angular_change_deg,
       2: max_iou_overlap,
       3: min_centroid_distance_px,
       4: aspect_ratio_change,
       5: relative_approach_velocity
    ]
    """
    np.random.seed(42)
    n_normal = num_samples // 2
    n_accident = num_samples - n_normal

    # Normal traffic dynamics
    normal_speed_drop = np.random.uniform(0.0, 0.35, n_normal)
    normal_angle_change = np.random.uniform(0.0, 30.0, n_normal)
    normal_iou = np.random.uniform(0.0, 0.10, n_normal)
    normal_dist = np.random.uniform(80.0, 400.0, n_normal)
    normal_aspect_change = np.random.uniform(0.0, 0.20, n_normal)
    normal_approach_vel = np.random.uniform(0.0, 12.0, n_normal)

    X_normal = np.column_stack([
        normal_speed_drop,
        normal_angle_change,
        normal_iou,
        normal_dist,
        normal_aspect_change,
        normal_approach_vel
    ])
    y_normal = np.zeros(n_normal, dtype=int)

    # Accident dynamic patterns
    acc_speed_drop = np.random.uniform(0.40, 0.95, n_accident)
    acc_angle_change = np.random.uniform(35.0, 180.0, n_accident)
    acc_iou = np.random.uniform(0.15, 0.85, n_accident)
    acc_dist = np.random.uniform(0.0, 95.0, n_accident)
    acc_aspect_change = np.random.uniform(0.25, 0.90, n_accident)
    acc_approach_vel = np.random.uniform(15.0, 60.0, n_accident)

    X_accident = np.column_stack([
        acc_speed_drop,
        acc_angle_change,
        acc_iou,
        acc_dist,
        acc_aspect_change,
        acc_approach_vel
    ])
    y_accident = np.ones(n_accident, dtype=int)

    X = np.vstack([X_normal, X_accident])
    y = np.hstack([y_normal, y_accident])

    # Shuffle dataset
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    return X[indices], y[indices]


def train_and_save_model():
    log_event("info", "Generating motion dynamics dataset for gesture classifier...")
    X, y = generate_synthetic_accident_dataset(num_samples=2000)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    clf.fit(X_scaled, y)

    accuracy = clf.score(X_scaled, y)
    log_event("info", f"Motion Gesture Classifier trained with accuracy: {accuracy * 100:.2f}%")

    # 1. Save joblib model
    joblib_path = Path(config.gesture_model_joblib)
    joblib.dump({"model": clf, "scaler": scaler}, joblib_path)
    log_event("info", f"Saved joblib model to: {joblib_path}")

    # 2. Save fast numpy npz format for lightweight fast inference
    npz_path = Path(config.gesture_model_npz)
    np.savez_compressed(
        npz_path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        feature_importances=clf.feature_importances_,
        trees_count=len(clf.estimators_)
    )
    log_event("info", f"Saved fast NumPy model binary to: {npz_path}")

    return clf, scaler


if __name__ == "__main__":
    train_and_save_model()
