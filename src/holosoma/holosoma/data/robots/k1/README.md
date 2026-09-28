# AI Sapiens K1 Rev.1 (23 DoF)

Robot description of the ROBOTIS AI Sapiens K1 Rev.1 humanoid used by
`holosoma.config_values.robot.k1_23dof`.

| File | Purpose |
| --- | --- |
| `k1_23dof.urdf` | IsaacGym / IsaacSim asset (USD is converted on first use) |
| `k1_23dof.xml` | MuJoCo asset (training and sim-to-sim) |
| `meshes/*.stl` | Visual meshes (millimetre STL, scaled 0.001 in the models) |

Both models are generated from the upstream `ai_sapiens_description` ROS package
(Apache-2.0, ROBOTIS CO., LTD.) by `scripts/build_k1_assets.py`. Changes versus upstream:

* `package://` mesh URIs are rewritten to `meshes/`.
* `left_foot_contact_point` / `right_foot_contact_point` bodies are added 65 mm below the
  ankle-roll frame (bottom of the foot collision spheres). holosoma uses them for foot
  height and as `key_bodies`, like the G1/T1 models.
* The fixed `head_link` is merged into `torso_link` in the MuJoCo model so the body list is
  identical across IsaacGym, IsaacSim (`collapse_fixed_joints=True`) and MuJoCo.
* MuJoCo actuators are named after their joints (holosoma resolves actuators by joint name)
  and the free joint is unnamed, matching the G1/T1 models.
* Hands are the compact rubber hand (shaft plus 30 mm ball). Their colliders are a capsule
  (x 0.03 to 0.15 m, radius 28 mm) and a sphere centred at (0.179, +/-0.0072, 0.0005) in the
  wrist-roll frame; the sphere centre is also the retargeting tool-center-point.

Joint order (also the order of `dof_names`, motion `joint_names` and the omni-k1 /
cyclo_lab exports):

```
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll,
right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll,
waist_yaw,
left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll,
right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_roll
```

Motion data for whole-body tracking is produced with
`src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_k1_motion.py`
and lives under `holosoma/data/motions/k1_23dof/whole_body_tracking/`.
