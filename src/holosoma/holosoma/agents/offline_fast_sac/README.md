# Offline FastSAC

기존 FastSAC의 Actor / distributional Critic 구조와 update equation을 그대로 유지하면서,
환경 rollout 대신 **정적 HDF5 데이터셋**으로 학습하는 오프라인 RL 트레이너입니다.

---

## 모듈 구성

| 파일 | 역할 |
|------|------|
| `offline_fast_sac_agent.py` | `OfflineFastSACAgent` — 오프라인 학습 루프, critic/actor/alpha 업데이트 |
| `offline_fast_sac_utils.py` | `OfflineReplayBuffer`, `validate_offline_dataset`, `init_normalizer_from_dataset` |

기존 코드에서 재사용하는 모듈:
- `fast_sac.py` → `Actor`, `Critic` (수정 없이 import)
- `fast_sac_utils.py` → `EmpiricalNormalization`, `save_params`
- `algo.py` → `FastSACConfig` (offline 필드 추가됨)

---

## 오프라인 데이터셋 스키마

HDF5 파일 (`offline_dataset.h5`)에 아래 키가 모두 포함되어야 합니다.

| 키 | Shape | dtype |
|----|-------|-------|
| `observations` | `[N, actor_obs_dim]` | float |
| `critic_observations` | `[N, critic_obs_dim]` | float |
| `actions` | `[N, action_dim]` | float |
| `rewards` | `[N]` 또는 `[N, 1]` | float |
| `next_observations` | `[N, actor_obs_dim]` | float |
| `next_critic_observations` | `[N, critic_obs_dim]` | float |
| `dones` | `[N]` 또는 `[N, 1]` | int/bool |
| `truncations` | `[N]` 또는 `[N, 1]` | int/bool |

- 모든 키의 첫 번째 차원 `N`은 동일해야 합니다.
- `rewards`, `dones`, `truncations`가 `[N, 1]`이면 자동으로 `[N]`으로 squeeze됩니다.

### 데이터셋 생성 예시

```python
import h5py
import numpy as np

N = 100000
actor_obs_dim = 48
critic_obs_dim = 96
action_dim = 12

with h5py.File("offline_dataset.h5", "w") as f:
    f.create_dataset("observations",            data=np.random.randn(N, actor_obs_dim).astype(np.float32))
    f.create_dataset("critic_observations",     data=np.random.randn(N, critic_obs_dim).astype(np.float32))
    f.create_dataset("actions",                 data=np.random.randn(N, action_dim).astype(np.float32))
    f.create_dataset("rewards",                 data=np.random.randn(N).astype(np.float32))
    f.create_dataset("next_observations",       data=np.random.randn(N, actor_obs_dim).astype(np.float32))
    f.create_dataset("next_critic_observations", data=np.random.randn(N, critic_obs_dim).astype(np.float32))
    f.create_dataset("dones",                   data=np.zeros(N, dtype=np.int64))
    f.create_dataset("truncations",             data=np.zeros(N, dtype=np.int64))
```

---

## Config 설정

`FastSACConfig`에 추가된 offline 전용 필드:

```python
FastSACConfig(
    # --- 필수 (offline_mode=True일 때) ---
    offline_mode=True,
    offline_dataset_path="/path/to/offline_dataset.h5",
    actor_obs_dim=48,
    critic_obs_dim=96,
    action_dim=12,

    # --- 선택 ---
    offline_action_scale=[],          # 비어 있으면 ones(action_dim). 관절별 스케일링.
    offline_normalizer_init_mode="dataset",  # "dataset" | "checkpoint" | "none"
    offline_num_updates_per_epoch=1000,      # epoch당 gradient step 수
    eval_interval=0,                  # 0이면 평가 안 함. 양수면 해당 epoch 간격으로 평가.
    conservative_weight=0.0,          # CODAC/SMQR 사용 시 양수로 설정

    # --- 기존 FastSAC 하이퍼파라미터 (그대로 사용 가능) ---
    num_learning_iterations=500,      # 총 epoch 수
    batch_size=8192,
    gamma=0.97,
    tau=0.125,
    policy_frequency=4,
    # ... 등등
)
```

### `offline_normalizer_init_mode` 옵션

| 값 | 동작 |
|----|------|
| `"dataset"` | 데이터셋 전체의 mean/std를 계산하여 normalizer에 설정 후 freeze |
| `"checkpoint"` | checkpoint에서 normalizer state를 로드 (`load()` 시 복원) |
| `"none"` | observation normalization 비활성화 (`nn.Identity`로 대체) |

---

## 사용 예시

```python
from holosoma.agents.offline_fast_sac.offline_fast_sac_agent import OfflineFastSACAgent
from holosoma.config_types.algo import FastSACConfig

config = FastSACConfig(
    offline_mode=True,
    offline_dataset_path="/data/offline_dataset.h5",
    actor_obs_dim=48,
    critic_obs_dim=96,
    action_dim=12,
    num_learning_iterations=500,
    offline_num_updates_per_epoch=1000,
    batch_size=4096,
    save_interval=100,
    logging_interval=10,
)

agent = OfflineFastSACAgent(
    config=config,
    device="cuda",
    log_dir="./logs/offline_fastsac",
    env=None,           # 평가 환경 (선택). None이면 평가 스킵.
)

agent.setup()
# agent.load("checkpoint.pt")  # 선택: 기존 checkpoint에서 이어서 학습
agent.learn()
```

---

## 체크포인트

- `save_interval` 간격으로 `model_XXXXXXX.pt`가 `log_dir`에 저장됩니다.
- 학습 종료 시 최종 checkpoint가 자동 저장됩니다.
- `agent.load("checkpoint.pt")`로 학습을 재개할 수 있습니다.
- Normalizer 상태도 checkpoint에 포함됩니다.

---

## 추론 (Inference)

```python
# 학습된 agent에서 policy 추출
policy_fn = agent.get_inference_policy(device="cuda")

# obs는 dict 형태: {"actor_obs": tensor}
action = policy_fn({"actor_obs": obs_tensor})
```

ONNX export도 지원됩니다:

```python
wrapper = agent.actor_onnx_wrapper
# torch.onnx.export(wrapper, dummy_input, "policy.onnx", ...)
```

---

## CODAC / SMQR 등 Conservative Loss 확장

`OfflineFastSACAgent`를 상속하고 hook 메서드를 override하세요:

```python
class CODACAgent(OfflineFastSACAgent):

    def _compute_conservative_critic_loss(self, data, q_outputs, critic_observations, actions):
        # q_outputs: [num_q, batch, num_atoms]
        # 여기에 conservative Q penalty 구현
        return conservative_penalty  # scalar tensor

    def _compute_conservative_actor_loss(self, data, actions, log_probs):
        # 필요 시 actor에 대한 conservative loss 추가
        return torch.tensor(0.0, device=self.device)
```

`config.conservative_weight > 0`을 설정하면 critic loss에 가중치가 곱해져 합산됩니다:

```
total_critic_loss = standard_distributional_loss + conservative_weight * conservative_critic_loss
```

---

## 주의사항

- `offline_mode=False`인 config로 `OfflineFastSACAgent`를 생성하면 `ValueError`가 발생합니다.
- 데이터셋의 `observations` 차원과 `actor_obs_dim`이 불일치하면 `AssertionError`가 발생합니다.
- 데이터셋은 전체가 GPU 메모리에 로드됩니다. 메모리 제약이 있을 경우 데이터셋 크기를 조절하세요.
- 기존 online FastSAC 코드는 전혀 수정되지 않았으므로 backward compatibility가 유지됩니다.
