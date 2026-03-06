# PhyNeSim — 输入数据结构

---

## 1. WIMUSim 四参数（B / D / P / H）

PhyNeSim 的物理仿真层接受四个参数对象，均为 `WIMUSim` 的内部类。

### B — Body（骨骼形状）

```python
WIMUSim.Body(
    rp: dict[str, torch.Tensor],   # 骨骼向量，key 为关节名，value shape (3,)
    device: torch.device,
)
```

`rp` 由 `compute_B_from_beta(betas, smpl_model_path)` 从 SMPL β 参数生成，
key 为 SMPL 关节名（如 `"pelvis"`, `"left_hip"`, ...），共 24 个关节。

---

### D — Dynamics（运动序列）

```python
WIMUSim.Dynamics(
    orientation: dict[str, torch.Tensor],  # 各关节四元数序列，shape (T, 4)，WXYZ 顺序
    translation: dict[str, torch.Tensor],  # 根节点平移，key 为 "XYZ"，shape (T, 3)，单位 m
    sample_rate: float,                    # 采样率，MoVi 为 120 Hz
    device: torch.device,
)
```

`orientation` 由 `smpl_pose_to_D_orientation(global_orient, body_pose)` 生成，
key 为 SMPL 关节名，共 24 个关节，每个 shape `(T, 4)`。

---

### P — Placement（传感器安装位置）

```python
WIMUSim.Placement(
    rp: dict[tuple, torch.Tensor],  # 传感器相对骨骼的位置偏移，shape (3,)
    ro: dict[tuple, torch.Tensor],  # 传感器相对骨骼的旋转偏移，shape (3, 3) 或 (4,)
    device: torch.device,
)
```

key 为 `(joint_name, imu_name)` 二元组，例如 `("left_forearm", "LLA")`。
由 `generate_default_placement_params(B_rp)` 生成，支持 15 个 IMU 位置（见下文）。

---

### H — Hardware（传感器噪声特性）

```python
WIMUSim.Hardware(
    ba: float,              # 加速度计偏置（bias）标准差
    bg: float,              # 陀螺仪偏置标准差
    sa: float,              # 加速度计比例因子误差
    sg: float,              # 陀螺仪比例因子误差
    sa_range_dict: dict,    # 各 IMU 加速度计量程
    sg_range_dict: dict,    # 各 IMU 陀螺仪量程
)
```

由 `wimusim.utils.generate_default_H_configs(imu_names)` 生成默认配置。

---

## 2. 支持的 IMU 位置

共 15 个（RSHO / LSHO 无默认安装参数，会被自动跳过）：

| 名称 | 身体位置 |
|------|---------|
| `HED` | 头部 |
| `STER` | 胸骨 |
| `PELV` | 骨盆 |
| `RUA` / `LUA` | 右 / 左上臂 |
| `RLA` / `LLA` | 右 / 左前臂 |
| `RHD` / `LHD` | 右 / 左手 |
| `RTH` / `LTH` | 右 / 左大腿 |
| `RSH` / `LSH` | 右 / 左小腿 |
| `RFT` / `LFT` | 右 / 左脚 |

---

## 3. 模型输入张量

`NeuralSimulator.forward(pose_6d, phys_imu)` 接受：

| 参数 | Shape | 说明 |
|------|-------|------|
| `pose_6d` | `(B, T, 24 × 6 = 144)` | 24 个 SMPL 关节的 6D 旋转表示，由四元数转换而来 |
| `phys_imu` | `(B, T, N × 6)` | WIMUSim 物理仿真输出，每个传感器 `[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]`，N 为有效传感器数 |

返回：

| 返回值 | Shape | 说明 |
|--------|-------|------|
| `corrected_imu` | `(B, T, N × 6)` | 修正后的虚拟 IMU（残差模式：`phys_imu + residual`） |

`B` = batch size，`T` = 时间窗口帧数（训练时默认 128），`N` = IMU 数量。

---

## 4. 四元数到 6D 旋转的转换

SMPL 关节方向以四元数 `(W, X, Y, Z)` 存储，输入模型前转换为 6D 旋转表示：

```python
# quat_wxyz_to_rot6d(q)：取旋转矩阵的前两列展平为 6 维向量
# q: (..., 4) WXYZ  →  rot6d: (..., 6)
```

6D 表示在 SO(3) 上连续，避免四元数的双覆盖问题（q 和 -q 代表同一旋转），
梯度更稳定（参考：Zhou et al., "On the Continuity of Rotation Representations in Neural Networks," CVPR 2019）。

---

## 5. 数据集输入格式（MoVi）

### F_amass_Subject_N.mat — SMPL 参数

| 字段 | Shape | 单位 / 说明 |
|------|-------|------------|
| `betas` | `(10,)` | SMPL 体型参数 β |
| `global_orient` | `(T, 3, 3)` | 根节点旋转矩阵 |
| `body_pose` | `(T, 23, 3, 3)` | 23 个身体关节旋转矩阵（不含根节点） |
| `trans` | `(T, 3)` | 根节点全局平移，单位 m |

采样率：120 Hz（MoVi AMASS）。

---

### imu_Subject_N.mat — Xsens 真实 IMU

| 字段 | Shape | 单位 / 说明 |
|------|-------|------------|
| 各传感器加速度 | `(T, 3)` | 单位 m/s² |
| 各传感器角速度 | `(T, 3)` | 单位 rad/s |

采样率：100 Hz，`load_xsens_imu()` 内部上采样至 120 Hz 以与 SMPL 对齐。
包含 17 个传感器位置（含 RSHO / LSHO）。

---

### SimulatorDataset 输出（训练用）

`__getitem__` 返回一个三元组，每个元素均为 `float32` Tensor：

| 变量 | Shape | 说明 |
|------|-------|------|
| `pose_6d` | `(T_win, 144)` | SMPL 关节 6D 旋转特征 |
| `phys_imu` | `(T_win, N × 6)` | WIMUSim 物理仿真 IMU |
| `real_imu` | `(T_win, N × 6)` | 真实 Xsens IMU（训练目标） |

`T_win` 默认 128 帧（≈ 1.07 s @ 120 Hz），`N` 为有效传感器数（最多 15）。
通道顺序（每个传感器）：`[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]`。
