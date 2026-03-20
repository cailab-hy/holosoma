import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


H5_PATH = "offline_data/fastsac_dataset.h5"   # 파일 경로로 바꾸세요
ACTION_KEY_CANDIDATES = [
    "actions",
    "action",
    "dataset/actions",
    "data/actions",
]


def print_h5_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"[DATASET] {name} | shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"[GROUP]   {name}")


def find_action_key(h5file):
    all_dataset_keys = []

    def collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            all_dataset_keys.append(name)

    h5file.visititems(collect)

    print("\n=== HDF5 내부 dataset 목록 ===")
    for k in all_dataset_keys:
        print("-", k)

    for cand in ACTION_KEY_CANDIDATES:
        if cand in all_dataset_keys:
            return cand

    for k in all_dataset_keys:
        low = k.lower()
        if "action" in low:
            return k

    return None


def analyze_actions(actions, save_prefix="action_stats"):
    actions = np.asarray(actions)

    if actions.ndim != 2:
        raise ValueError(f"actions shape이 2차원이 아닙니다: {actions.shape}")

    n, d = actions.shape
    print(f"\nactions shape: {actions.shape}")

    a_min = actions.min(axis=0)
    a_max = actions.max(axis=0)
    a_mean = actions.mean(axis=0)
    a_std = actions.std(axis=0)
    a_p1 = np.percentile(actions, 1, axis=0)
    a_p99 = np.percentile(actions, 99, axis=0)

    df = pd.DataFrame({
        "dim": np.arange(d),
        "min": a_min,
        "max": a_max,
        "mean": a_mean,
        "std": a_std,
        "p1": a_p1,
        "p99": a_p99,
    })

    print("\n=== action dimension별 통계 ===")
    print(df.to_string(index=False))

    csv_path = f"{save_prefix}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 저장 완료: {csv_path}")

    rows = int(np.ceil(d / 6))
    cols = min(6, d)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(d):
        axes[i].hist(actions[:, i], bins=50)
        axes[i].set_title(f"action dim {i}")
        axes[i].grid(alpha=0.3)

    for j in range(d, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig_path = f"{save_prefix}_hist.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"히스토그램 저장 완료: {fig_path}")


def main():
    with h5py.File(H5_PATH, "r") as f:
        print("=== HDF5 구조 ===")
        f.visititems(print_h5_structure)

        action_key = find_action_key(f)
        if action_key is None:
            raise KeyError(
                "actions dataset을 찾지 못했습니다. "
                "출력된 dataset 목록을 보고 action key를 직접 지정하세요."
            )

        print(f"\n사용할 action key: {action_key}")
        actions = f[action_key][:]

    analyze_actions(actions, save_prefix="action_stats")


if __name__ == "__main__":
    main()