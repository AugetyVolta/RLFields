# PGRL-Nav

多机器人导航训练与评测代码。当前统一使用标准模块和共享策略，不需要选择不同版本。

- 使用 **Ray/RLlib 2.51.1，Policy + ModelV2 旧 API 栈**。
- 仿真使用 **IR-SIM 2.12.1 `[fast]`**，启用新版批量 LiDAR 和 numba 编译内核。
- 默认训练 **10 个机器人、200 轮**；机器人半径 **0.1 m**，到达阈值 **0.3 m**。
- SDF 梯度、context FFN、邻居 self-attention 默认开启；梯度缩放为 `8.0`、裁剪为 `1.0`。
- 新训练必须指定环境目录；续训和评测读取已保存的实验配置。

## 1. 安装与运行位置

```bash
conda create -n pgrl python=3.10
conda activate pgrl
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

`requirements.txt` 已固定 `ir-sim[fast]==2.12.1`；使用官方发行包，不再安装原来的自定义 IR-SIM 分支。`[fast]` 会安装 numba/llvmlite，第一次启动需要编译 LiDAR 内核，后续使用缓存。比较吞吐量时应先预热。

迁移已准备独立的本地运行环境，不改动原 Conda 环境。当前机器可以直接使用：

```bash
source ~/.venvs/pgrl-irsim2121/bin/activate
```

这个本地环境复用原 Conda 环境中的基础依赖，单独安装新仿真器和 Ray。全新机器请使用上面的 Conda 安装步骤。

以下命令均在项目根目录执行：

```bash
cd ~/RLFields
```

训练入口按 GPU 训练配置构建；评测可通过 `--device cpu` 在 CPU 上运行。

## 2. 环境目录：地图与规划器权重一起选

`--env` 填写的是 **`config/` 下的文件夹名**。例如：

```text
config/
  random_env/
    random_env.yaml
    random_env.obj
    random_env.pt
  random_env_1/
    random_env_1.yaml
    random_env_1.obj
    random_env_1.pt
  random_env_2/
    random_env_2.yaml
    random_env_2.obj
    random_env_2.pt
  my_map/
    scene.yaml
    geometry.obj
    planner_weights.pt
```

```bash
python train.py --env random_env_1
python train.py --env my_map
```

目录名可以自行定义，没有写死环境名单。每个目录需要：

- **一个环境 YAML**：`.yaml` 或 `.yml`，内容同时包含 `world` 和 `robot`。
- **一份对应地图的 PC-Planner 权重**：`.pt`、`.pth` 或 `.ckpt` 文件。
- 可以附带 `.obj`、不含上述环境结构的参数快照等文件。

文件名不要求与目录名相同，YAML 和权重也不要求同名。缺少候选或存在多份候选都会报错，不会猜选。没有默认环境，也不会退回根目录寻找默认权重。

当前三个环境的 YAML、OBJ 和预训练 `.pt` 权重均已随仓库保存。新增环境时，将对应地图和为其训练的 PC-Planner 权重一起放入新目录；选择目录会同时选中地图和权重，仍需保证两者确实匹配。

## 3. 参数在何处修改

默认参数集中在 `env/params.py`。新训练可以修改默认值，也可以用 `--override-config` 传入只包含改动项的 YAML。

| 参数分组 | 可修改内容 | 当前主要默认值 |
| --- | --- | --- |
| `robot_params` | 机器人数量、半径、到达阈值 | `10`、`0.1 m`、`0.3 m` |
| `robot_params` | LiDAR 最小/最大距离、扫描角度、束数 | `0.0 / 6.0 m`、`6.28 rad`、`200` |
| `robot_params` | 单机器人最大任务步数、最大速度 | `500`、`1.5 m/s` |
| `scene_params` | `step_time`、`sample_time`、`collision_mode` | `0.1 s`、`null`、`stop` |
| `experiment_params` | 总轮次、绘图、采样进程及资源配置 | 总轮次 `200`；资源见下方示例 |
| `algorithm_params` | PPO 学习率、batch、epochs 等 | `lr=5e-5`、batch `16000`、epochs `15` |
| `network_params` / `esdf_params` / `reward_params` | 网络、局部几何与奖励参数 | 以代码中的默认值为准 |
| `map_generation_params` | 新地图尺寸、障碍物数量、间距及尺寸范围 | 仅生成新地图时使用 |

机器人和仿真时间参数在创建 IR-SIM 前覆盖地图 YAML，**不用同时修改地图文件**。`sample_time: null` 表示跟随 `step_time`；显式设置时必须是 `step_time` 的正整数倍。动作位移尺度直接使用 `step_time`。

已有地图的大小、偏移和障碍物布局从所选 YAML 读取。当前机器人固定为 `omni`、`circle`，不再单独配置。实际地图信息、固定的 `step_mode: external` 和自动解析的路径会导出到 `scene_config.yaml` 的 `environment_info` 等字段；实际仿真配置归档为 `env.yaml`。

### 按本次实验覆盖参数

例如，创建 `run_override.yaml`：

```yaml
robot_params:
  number_robots: 10
  radius: 0.1
  goal_threshold: 0.3
scene_params:
  step_time: 0.1
  sample_time: null
experiment_params:
  disable_plotting: true
  max_iters: 200
  num_env_runners: 8
  num_envs_per_env_runner: 4
  num_cpus_per_env_runner: 3
  num_gpus_per_env_runner: 0.25
  num_cpus_for_main_process: 4
algorithm_params:
  lr: 0.00005
```

```bash
python train.py --env random_env_1 --override-config run_override.yaml
```

上面是可选的服务器配置示例。代码当前资源默认值为：`num_env_runners=2`、每进程一个环境、每进程 `10` CPU / `0.5` GPU、主进程 `4` CPU。

**现在可以设置 `num_envs_per_env_runner=4`**：已验证同一 Ray 采样进程内 4 个环境、每个环境 10 个机器人，机器人编号、状态、激光及清理互相隔离。总采样环境数为 `num_env_runners × num_envs_per_env_runner`；CPU/GPU 配额按采样进程分配，无须为其中每个环境重复申请一份。环境在该进程中按序推进，策略推理可以合批；增加环境数不等于 CPU 步进同时并行，也不保证同比例加速。按服务器资源和实测吞吐量调整配置。

服务器训练时使用 `disable_plotting: true`，会创建真正的 headless 环境；默认参数中仍保留原有绘图设置。

只写需要覆盖的项即可。未知分组、拼错字段或尝试覆盖自动生成的路径会报错。`environment_info` 是导出信息，不是覆盖参数。修改网络结构或观测维度后，原 checkpoint 可能无法恢复。

## 4. 新训练与实验布局

```bash
python train.py --env random_env
python train.py --env random_env_1 --max-iters 200
python train.py --env random_env_2 --output-dir logs/my_experiment
```

未指定输出目录时，自动生成：

```text
logs/multi_robot_ppo_<环境目录名>_<年月日>_<时分秒>_<微秒>/
```

时间使用 `Asia/Shanghai`。`--output-dir` 表示完整实验目录，不是其父目录；新目录应不存在或为空。单个实验沿用原 NAS 日志布局：

```text
logs/<实验名>/
  scene_config.yaml             # 新训练或新目录续训时的配置
  scene_config_resume.yaml      # 原目录续训时保存的本次配置
  env.yaml                     # 实际生效的地图与仿真参数
  ntfields.pt                  # 本次使用的 PC-Planner 权重
  ckpts/
    iter_000001/
    iter_000002/
    ...
    final/
  tensorboard/
    events.out.tfevents.*
    params.json
    params.pkl
    progress.csv
    result.json
```

每完成一轮保存一个 `iter_XXXXXX` checkpoint，训练结束额外保存 `final/`。日志中的轮次从 `1` 开始，与 checkpoint 目录一致。当前续训与评测入口均通过数字轮次选择 `iter_XXXXXX`。

查看训练曲线：

```bash
tensorboard --logdir logs
```

## 5. 续训：先恢复原配置，再覆盖明确指定的项

在原实验目录从第 100 轮继续到总计 200 轮：

```bash
python train.py --resume logs/my_experiment --iter 100 --max-iters 200
```

保留原实验，将后续训练写入另一个目录：

```bash
python train.py --resume logs/my_experiment --iter 100 --max-iters 200 --output-dir logs/my_experiment_continued
```

从已完成的 200 轮继续到 300 轮，并覆盖部分参数：

```bash
python train.py --resume logs/my_experiment --iter 200 --max-iters 300 --override-config resume_override.yaml
```

例如 `resume_override.yaml`：

```yaml
experiment_params:
  disable_plotting: true
  num_env_runners: 8
  num_cpus_per_env_runner: 3
algorithm_params:
  lr: 0.00002
```

续训规则：

- `--resume` 指定实验目录，`--iter N` 选择 `ckpts/iter_XXXXXX`，从 **N+1** 轮继续；不能同时指定 `--env`。
- 恢复模型、优化器和训练计数，并核对 checkpoint 实际轮次。`--max-iters` 是**总轮次上限**，不是额外轮数，必须大于 N。
- 配置优先级：**checkpoint 保存的配置 → `--override-config` 中明确写出的字段 → `--max-iters`**。不把当前 `params.py` 的默认值整套覆盖到旧实验上。
- 默认在原目录追加日志，保留 `scene_config.yaml`，本次配置写入 `scene_config_resume.yaml`。更换输出目录时，会归档一套独立的地图、规划器权重和配置。
- 原目录存在即将写入的后续轮次 checkpoint 时会拒绝覆盖，请用 `--output-dir` 分支续训。新输出目录应不存在或为空。
- 优先使用实验目录归档的 `env.yaml`、`ntfields.pt`。缺少归档时尝试保存的原路径；仍找不到则报错，不会换成其他默认资产。

未内嵌完整配置的历史 checkpoint 会尝试读取实验目录中的配置快照。仍引用已移除模块名的 checkpoint，先按第 8 节迁移。

## 6. 评测

两个入口都通过**训练实验目录 + 轮次**加载模型和训练配置，无需再次选择地图。

### 运行或显示规划过程

```bash
python evaluate.py --log-dir logs/my_experiment --iter 200 --episodes 2
python evaluate.py --log-dir logs/my_experiment --iter 200 --num-robots 20 --episodes 2 --max-steps 5000 --render
```

默认不绘图，`--render` 开启显示。CPU 评测示例：

```bash
python evaluate.py --log-dir logs/my_experiment --iter 200 --device cpu --cpus 4
```

### 汇总指标并导出 JSON

```bash
python evaluate_metrics.py --log-dir logs/my_experiment --iter 200 --num-robots 20 --episodes 100 --max-steps 5000 --out-json result.json
```

JSON 包含逐回合的 `episodes`、总体 `summary` 和实际运行参数 `evaluation_config`，记录训练机器人数量、实际评测数量、轮次、回合数、步数上限和设备。指标包括团队/单机器人成功率、路径长度、回报、速度、步数及奖励分项；计算方式沿用现有评测实现。

| 参数 | 含义与默认值 |
| --- | --- |
| `--log-dir` | 必填，训练实验目录 |
| `--iter` | 必填，正整数，选择已保存的训练轮次 |
| `--num-robots` | 可选，覆盖实际仿真机器人数量；不传则保持训练数量 |
| `--episodes` | 回合数；`evaluate.py` 默认 `2`，`evaluate_metrics.py` 默认 `20` |
| `--max-steps` | 每回合外层推进上限，默认 `5000`；训练配置中的单机器人超时仍生效 |
| `--cpus` | Ray 逻辑 CPU 配额，默认 `8` |
| `--device` | `cpu` 或 `cuda`；检测到可用 CUDA 时默认 `cuda`，否则 `cpu` |
| `--render` | 可选，开启显示 |
| `--out-json` | 仅指标入口支持，可选，导出 JSON 到指定文件 |

`--log_dir`、`--num_robots`、`--max_steps`、`--out_json` 也保留为对应参数的别名。

评测使用主进程中的一个环境，`num_env_runners=0` 表示不创建远程采样进程。`--num-robots` 会改变真实仿真对象数量。评测不改写训练配置、原地图或 checkpoint；终端会显示实际运行参数。

## 7. 生成新随机地图

在 `env/params.py` 的 `MapGenerationParameters` 中保留了地图宽高、障碍物数量、最小间距、圆形半径范围和矩形边长范围。它们只影响新地图生成，不改变已经选中的地图。

```bash
python -m utils.gen_env
```

当前示例入口输出 `config/random_env.yaml` 和 `config/random_env.obj`，不是直接写入已有环境子目录。生成后，将所需地图文件放入一个新的 `config/<环境目录名>/`，再加入为该地图训练的 PC-Planner 权重，才能通过 `--env` 用于导航训练。本仓库的这个生成入口不负责训练规划器。

## 8. 迁移历史 checkpoint

训练和评测只使用当前标准模块与 `rlfields_actor` 注册名。旧模型如果引用已移除的模块，使用独立工具生成可加载的副本：

```bash
python tools/migrate_checkpoint.py --checkpoint /path/to/old/checkpoint --output-dir /path/to/new/experiment
```

`--checkpoint` 指向完整的 RLlib Algorithm checkpoint 目录，需包含 `shared_policy`；`--output-dir` 应不存在或为空。工具保留权重、优化器状态和实际迭代计数，复制地图和规划器权重，生成标准实验布局，打印对应续训命令。原 checkpoint、配置和日志不修改。

如果历史奖励字段改过名字，工具会列出未识别的键，用 `--rename-key OLD=NEW` 明确指定映射，可以重复传入。例如当前方向奖励权重名称为 `w_heading`，对应指标为 `r_heading`、`r_heading_raw`。工具只迁移模块命名和配置格式，不转换不兼容的网络结构。

迁移后的续训仍遵循总轮次规则：若 checkpoint 已完成 200 轮，继续训练时需要指定如 `--max-iters 300`。

## 9. 已完成的功能验证

原版已完成新训练、原目录/新目录续训、两个评测入口，以及历史模型命名迁移的短流程验证。本次 IR-SIM 迁移另外在 Ray 2.51.1 旧栈、CPU 条件下验证了：

- 同进程 4 个环境、每环境 10 个机器人：编号、位置、LiDAR、随机数生成器隔离；创建或关闭其他环境不影响存活环境。
- 真实 RLlib 采样进程使用 4 个环境完成 PPO 更新；旧版 checkpoint 恢复时权重、优化器状态一致，续训后优化器步数继续增长。
- 两个评测入口正常加载旧版/新版实验，实际切换为 3 / 20 个机器人，并导出指标 JSON。
- `step_time=0.1 / 0.2` 的位移尺度、停止、到达、机器人碰撞与障碍物碰撞。

配对固定动作回放结果如下，均为单线程 CPU、Ray 2.51.1、200 束 LiDAR；每种规模 2 组任务，每组预热 4 步、计时 12 步，表中是两组平均耗时的中位数：

| 完整采样计算回放 | 旧自定义 IR-SIM 2.7.5 | 新 IR-SIM 2.12.1 fast | 加速 |
| --- | --- | --- | --- |
| 10 个机器人 | 296.9 ms/世界步 | 85.0 ms/世界步 | 3.49 倍 |
| 20 个机器人 | 590.9 ms/世界步 | 171.3 ms/世界步 | 3.45 倍 |

所有计时步处理全部机器人，包含真实冻结策略前向、仿真、LiDAR、规划场和 UDF，不包含 PPO 更新或 Ray 通信。在这些配对轨迹上，位置、观测、策略输出、奖励和终止标记完全一致，测距误差小于 `1.4e-14 m`。这不能直接推算整轮训练或 GPU 吞吐量的加速，也不代表重跑了完整 200 轮收敛实验。

## 10. 新仿真器如何兼容原模型

旧分支的动作实际是已经乘过 `dt` 的世界坐标位移；新版默认接收速度。当前通过官方 `step_mode: external` 写入同样的位移，再由新版同步几何、传感器、碰撞和时间，避免动作被重复乘以 `dt`。

项目内 `env/sim/irsim_interface.py` 保留原任务采样所需的几何和起终点间距约束，采样只读取本环境对象和独立 RNG。策略点云仍按旧版的角度增量构造，实际射线和测距使用新版加速实现。这样保留 PGRL 的模型结构与输入接口，可以继续使用原规划器权重及兼容的策略 checkpoint。

当前任务沿用 `omni`、`circle` 和原碰撞判定。external 模式由项目负责运动更新；它不执行新版 `contact` 刚体动力学，也不自动推进配置中的自主运动障碍物。
