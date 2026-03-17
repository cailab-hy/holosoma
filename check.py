import h5py
import numpy as np


EXPECTED_KEYS = [
    "observations",
    "actions",
    "critic_observations",
    "next_observations",
    "next_critic_observations",
    "rewards",
    "truncations",
    "dones",
]


def _first_dim(x):
    return x.shape[0]


def _feature_shape(x):
    return x.shape[1:]


def _squeeze_binary(arr):
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] == 1:
            arr = arr[:, 0]
    return arr


def _scan_stats(ds, chunk_rows=8192):
    total = ds.shape[0]
    nan_count = 0
    inf_count = 0
    finite_count = 0
    s = 0.0
    mn = None
    mx = None

    for start in range(0, total, chunk_rows):
        end = min(total, start + chunk_rows)
        x = ds[start:end]
        x = np.asarray(x)

        if np.issubdtype(x.dtype, np.number):
            nan_mask = np.isnan(x)
            inf_mask = np.isinf(x)
            finite_mask = np.isfinite(x)

            nan_count += int(nan_mask.sum())
            inf_count += int(inf_mask.sum())

            finite_vals = x[finite_mask]
            if finite_vals.size > 0:
                finite_count += int(finite_vals.size)
                s += float(finite_vals.astype(np.float64).sum())
                cur_mn = float(finite_vals.min())
                cur_mx = float(finite_vals.max())
                mn = cur_mn if mn is None else min(mn, cur_mn)
                mx = cur_mx if mx is None else max(mx, cur_mx)

    mean = s / finite_count if finite_count > 0 else None
    return {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "finite_count": finite_count,
        "min": mn,
        "max": mx,
        "mean": mean,
    }


def _chunk_temporal_consistency(
    obs_ds,
    next_obs_ds,
    dones_ds,
    trunc_ds,
    num_envs,
    atol=1e-6,
    chunk_steps=128,
):
    total = obs_ds.shape[0]
    assert total % num_envs == 0, f"total samples {total} is not divisible by num_envs {num_envs}"
    num_steps = total // num_envs

    valid_count = 0
    valid_close_count = 0
    valid_max_abs = 0.0

    done_only_count = 0
    done_only_close_count = 0
    done_only_max_abs = 0.0

    for step_start in range(0, num_steps - 1, chunk_steps):
        step_end = min(num_steps - 1, step_start + chunk_steps)

        row_a0 = step_start * num_envs
        row_a1 = step_end * num_envs
        row_b0 = (step_start + 1) * num_envs
        row_b1 = (step_end + 1) * num_envs

        a = np.asarray(next_obs_ds[row_a0:row_a1], dtype=np.float32)
        b = np.asarray(obs_ds[row_b0:row_b1], dtype=np.float32)

        d = _squeeze_binary(dones_ds[row_a0:row_a1]).astype(np.int64)
        t = _squeeze_binary(trunc_ds[row_a0:row_a1]).astype(np.int64)

        a = a.reshape(a.shape[0], -1)
        b = b.reshape(b.shape[0], -1)

        per_row_max_abs = np.max(np.abs(a - b), axis=1)

        valid_mask = (d == 0) & (t == 0)
        done_only_mask = (d != 0) & (t == 0)

        if np.any(valid_mask):
            vd = per_row_max_abs[valid_mask]
            valid_count += vd.shape[0]
            valid_close_count += int((vd <= atol).sum())
            valid_max_abs = max(valid_max_abs, float(vd.max()))

        if np.any(done_only_mask):
            dd = per_row_max_abs[done_only_mask]
            done_only_count += dd.shape[0]
            done_only_close_count += int((dd <= atol).sum())
            done_only_max_abs = max(done_only_max_abs, float(dd.max()))

    return {
        "num_steps": num_steps,
        "valid_count": valid_count,
        "valid_close_count": valid_close_count,
        "valid_close_ratio": (valid_close_count / valid_count) if valid_count > 0 else None,
        "valid_max_abs": valid_max_abs,
        "done_only_count": done_only_count,
        "done_only_close_count": done_only_close_count,
        "done_only_close_ratio": (done_only_close_count / done_only_count) if done_only_count > 0 else None,
        "done_only_max_abs": done_only_max_abs,
    }


def validate_h5(path, num_envs=None, atol=1e-6):
    print("=" * 80)
    print(f"validate_h5: {path}")
    print("=" * 80)

    with h5py.File(path, "r") as f:
        print("\n[1] key 존재 여부")
        keys = list(f.keys())
        print("keys:", keys)

        missing = [k for k in EXPECTED_KEYS if k not in f]
        extra = [k for k in keys if k not in EXPECTED_KEYS]

        if missing:
            print("  - missing keys:", missing)
        else:
            print("  - required keys: OK")

        if extra:
            print("  - extra keys:", extra)

        if missing:
            print("\n필수 key가 없어서 이후 검사를 중단합니다.")
            return

        print("\n[2] 첫 축 길이 / shape 검사")
        shapes = {k: f[k].shape for k in EXPECTED_KEYS}
        dtypes = {k: f[k].dtype for k in EXPECTED_KEYS}

        for k in EXPECTED_KEYS:
            print(f"  - {k:24s} shape={shapes[k]} dtype={dtypes[k]}")

        lengths = {k: _first_dim(f[k]) for k in EXPECTED_KEYS}
        unique_lengths = sorted(set(lengths.values()))
        if len(unique_lengths) == 1:
            print("  - first dimension length: OK")
        else:
            print("  - first dimension mismatch:", lengths)

        obs_shape = _feature_shape(f["observations"])
        next_obs_shape = _feature_shape(f["next_observations"])
        critic_shape = _feature_shape(f["critic_observations"])
        next_critic_shape = _feature_shape(f["next_critic_observations"])

        print("\n[3] feature shape 검사")
        print(f"  - observations vs next_observations: {obs_shape} vs {next_obs_shape}")
        print(f"  - critic_observations vs next_critic_observations: {critic_shape} vs {next_critic_shape}")

        if obs_shape == next_obs_shape:
            print("  - actor obs shape pair: OK")
        else:
            print("  - actor obs shape pair: MISMATCH")

        if critic_shape == next_critic_shape:
            print("  - critic obs shape pair: OK")
        else:
            print("  - critic obs shape pair: MISMATCH")

        print("\n[4] NaN / Inf / basic stats")
        for k in EXPECTED_KEYS:
            if np.issubdtype(f[k].dtype, np.number):
                st = _scan_stats(f[k])
                print(
                    f"  - {k:24s} "
                    f"nan={st['nan_count']} inf={st['inf_count']} "
                    f"min={st['min']} max={st['max']} mean={st['mean']}"
                )

        print("\n[5] dones / truncations 논리 검사")
        dones = _squeeze_binary(f["dones"][:]).astype(np.int64)
        truncs = _squeeze_binary(f["truncations"][:]).astype(np.int64)

        if dones.shape[0] != truncs.shape[0]:
            print("  - dones/truncations 길이 불일치")
        else:
            invalid_trunc = int(((truncs != 0) & (dones == 0)).sum())
            done_count = int((dones != 0).sum())
            trunc_count = int((truncs != 0).sum())

            print(f"  - done count: {done_count}")
            print(f"  - truncation count: {trunc_count}")
            print(f"  - truncation인데 done=0 인 개수: {invalid_trunc}")

            if invalid_trunc == 0:
                print("  - truncation => done 관계: OK")
            else:
                print("  - truncation => done 관계: CHECK NEEDED")

        if num_envs is not None:
            print("\n[6] 시간축 정합성 검사")
            total = f["observations"].shape[0]
            if total % num_envs != 0:
                print(f"  - total={total} 가 num_envs={num_envs} 로 나누어 떨어지지 않습니다.")
            else:
                actor_tc = _chunk_temporal_consistency(
                    f["observations"],
                    f["next_observations"],
                    f["dones"],
                    f["truncations"],
                    num_envs=num_envs,
                    atol=atol,
                )
                critic_tc = _chunk_temporal_consistency(
                    f["critic_observations"],
                    f["next_critic_observations"],
                    f["dones"],
                    f["truncations"],
                    num_envs=num_envs,
                    atol=atol,
                )

                print(f"  - num_steps: {actor_tc['num_steps']}")
                print("  - actor non-terminal consistency")
                print(f"      valid_count      = {actor_tc['valid_count']}")
                print(f"      close_count      = {actor_tc['valid_close_count']}")
                print(f"      close_ratio      = {actor_tc['valid_close_ratio']}")
                print(f"      max_abs_diff     = {actor_tc['valid_max_abs']}")

                print("  - critic non-terminal consistency")
                print(f"      valid_count      = {critic_tc['valid_count']}")
                print(f"      close_count      = {critic_tc['valid_close_count']}")
                print(f"      close_ratio      = {critic_tc['valid_close_ratio']}")
                print(f"      max_abs_diff     = {critic_tc['valid_max_abs']}")

                print("  - done=1, trunc=0 에서 next_obs[t] 와 obs[t+1] 비교")
                print("    이 값이 너무 높으면 terminal obs 대신 reset obs를 저장했을 가능성이 있습니다.")
                print(f"      done_only_count  = {actor_tc['done_only_count']}")
                print(f"      close_count      = {actor_tc['done_only_close_count']}")
                print(f"      close_ratio      = {actor_tc['done_only_close_ratio']}")
                print(f"      max_abs_diff     = {actor_tc['done_only_max_abs']}")

        else:
            print("\n[6] 시간축 정합성 검사는 num_envs가 있어야 합니다.")
            print("    예: validate_h5(path, num_envs=4096)")

        print("\n[7] 판정 가이드")
        print("  - 모든 key의 첫 축 길이가 같아야 합니다.")
        print("  - observations 와 next_observations 의 feature shape는 같아야 합니다.")
        print("  - critic_observations 와 next_critic_observations 의 feature shape도 같아야 합니다.")
        print("  - NaN/Inf 는 보통 0이어야 합니다.")
        print("  - non-terminal consistency close_ratio 는 거의 1.0 이어야 정상입니다.")
        print("  - done=1,trunc=0 의 close_ratio 가 높으면 reset obs 저장 가능성을 의심하세요.")


if __name__ == "__main__":
    # 예시
    validate_h5(
        path="offline_data/fastsac_dataset.h5",
        num_envs=4,   # 네가 실제로 쓴 env.num_envs 로 바꿔
        atol=1e-6,
    )