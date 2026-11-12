# PGRL-Nav
### Intall
```
conda create -n irsim python=3.10
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 默认实现
训练、续训和评测默认使用 prior_align，不需要再选择版本。
SDF 梯度、context FFN、邻居 self-attention 默认开启，梯度缩放为 8.0、裁剪为 1.0。
Ray 保持 2.51.1，使用 Policy + ModelV2 旧 API 栈。

### 训练与评测
```
cd ~/RLFields
python train.py
python train.py --ckpt /path/to/checkpoint
python evaluate.py --ckpt /path/to/checkpoint --render
python evaluate_metrics.py --ckpt /path/to/checkpoint --episodes 100 --out_json result.json
```
评测和续训自动读取 checkpoint 保存时的增强配置；旧 checkpoint 不需要手动修改这几个开关。
默认训练仍为 10 个机器人、200 轮。默认地图和 PC-Planner 权重放在 config/random_env.yaml、ckpt.pt。
原来带 prior 后缀的文件已合并到标准文件中，旧 checkpoint 的类名在读取时自动映射。
当前代码沿用你的自定义 IR-SIM 分支，需要支持 get_random_free_position 的 filter_start_goal 参数；本次未更换仿真接口。
