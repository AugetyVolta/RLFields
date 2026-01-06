import irsim
import numpy as np
import torch

from model.models import NN
from .esdf_2d_simple import ESDF_2d
from ..params import SceneConfig
from utils.utils import lidar_to_pointcloud
from .sim_env import SIM_ENV
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class MARL_SIM(SIM_ENV):
    def __init__(
        self,
        world_file="config/random_env.yaml",
        disable_plotting=False,
        scene_cfg = SceneConfig()
    ):
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        self.num_robots = len(self.env.robot_list)
        self.robot_list = self.env.robot_list
        self.x_range = self.env._world.x_range
        self.y_range = self.env._world.y_range
        self.world_w = self.x_range[1] - self.x_range[0]
        self.world_h = self.y_range[1] - self.y_range[0]
        self.world_center = np.array(
            [(self.x_range[0] + self.x_range[1]) / 2.0,
            (self.y_range[0] + self.y_range[1]) / 2.0],
            dtype=np.float32,
        )
        # 用较大的那一边做统一尺度
        self.world_scale = max(self.world_w, self.world_h)

        self.self_obs_dim = scene_cfg.network_params.self_obs_dim  # [x, y, goal_x, goal_y]
        self.neighbor_obs_dim = scene_cfg.network_params.neighbor_obs_dim  # [x, y, goal_x, goal_y] for each neighbor robot

        self.esdf_cfg = scene_cfg.esdf_params
        self.esdf = ESDF_2d(self.esdf_cfg)
        self.r_max = self.esdf_cfg.r_max
        self.num_use_neighbor_obs = scene_cfg.network_params.num_use_neighbor_obs  

        self.goal_threshold = scene_cfg.robot_params.goal_threshold

        self.single_max_timestep = scene_cfg.robot_params.single_max_timestep 
        self.move_single_step = scene_cfg.robot_params.move_single_step  
        self.v_max = scene_cfg.network_params.nt_vmax 
        
        self.robot_times = torch.zeros(self.num_robots, dtype=torch.int32)  # time step counter
        self.robot_terminateds = torch.zeros(self.num_robots, dtype=torch.bool) # terminate
        self.robot_truncateds = torch.zeros(self.num_robots, dtype=torch.bool) # truncate

        self.robot_last_goal_dist = torch.zeros(self.num_robots, dtype=torch.float32)
        self.robot_radius = scene_cfg.robot_params.radius
        self.robot_last_action = np.zeros((self.num_robots, 2), dtype=np.float32)

        self.device = scene_cfg.network_params.device
        self.ntfields_nn = NN(device=scene_cfg.network_params.device,
                            dim=scene_cfg.network_params.nt_dim,
                            vmax=scene_cfg.network_params.nt_vmax,
                            h_size=scene_cfg.network_params.nt_h_size,
                            enc_d=scene_cfg.network_params.nt_enc_d,
                            dec_d=scene_cfg.network_params.nt_dec_d,
                            full_coords_enc=scene_cfg.network_params.nt_full_coords_enc,
                            basic_block=scene_cfg.network_params.nt_basic_block,
                            use_unet=scene_cfg.network_params.nt_use_unet,
                            symm_op_type=scene_cfg.network_params.nt_symm_op_type).to(self.device)
        
        nt_pretrained_path = scene_cfg.network_params.nt_pretrained_path
        ckpt = torch.load(nt_pretrained_path, map_location=self.device)
        self.ntfields_nn.load_state_dict(ckpt["model_state_dict"], strict=True)

        self.robot_last_T = torch.zeros(self.num_robots, dtype=torch.float32)
        self.robot_last_peer_phi = torch.zeros(self.num_robots, dtype=torch.float32)
        self.robot_last_obst_loss = torch.zeros(self.num_robots, dtype=torch.float32)

        self._env_step_id = 0
    # only alive robots have action
    def step(self, action_dict): # action_dict: dict of(robot_id, [d_x, d_y]), [-1, 1]
        applied_actions = []
        action_ids = []
        raw_actions = {}  # 新增：记录归一化动作，用于平滑/速度惩罚
        for robot_id, action in action_dict.items():
            id = int(robot_id.split('_')[-1]) # 'robot_0' -> 0
            action_ids.append(id)
            # 记录原始归一化动作
            raw_actions[id] = np.array(action, dtype=np.float32)
            d_x, d_y = action
            d_x *= self.world_w * self.move_single_step / self.v_max
            d_y *= self.world_h * self.move_single_step / self.v_max
            applied_actions.append([d_x, d_y])

        self.env.step(action=applied_actions, action_id=action_ids)
        self.env.render()
        self._env_step_id += 1

        obs = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}

        for id in action_ids:
            robot_id = f'robot_{id}'
            
            # 检查是否终止
            terminateds[robot_id], is_collision, is_arrive = self._is_terminated(id)
            
            # 更新时间步计数器, 检查是否截断
            truncateds[robot_id] = False
            is_truncated = False
            if not self.robot_terminateds[id]:
                self.robot_times[id] += 1
                if self.robot_times[id] >= self.single_max_timestep:
                    self.robot_truncateds[id] = True
                    is_truncated = True
                    truncateds[robot_id] = self.robot_truncateds[id].item()
            
            # 准备观测和info
            obs[robot_id] = self._prepare_obs(id)
            infos[robot_id] = {}

            # 计算 reward + info
            r, info = self._calculate_reward(
                id,
                raw_actions[id],
                obs[robot_id],
                is_collision,
                is_arrive,
                is_truncated,
            )
            rewards[robot_id] = r
            infos[robot_id].update(info)
            infos[robot_id]["_env_step_id"] = self._env_step_id
        
        # 处理所有机器人终止的情况
        robot_done = self.robot_terminateds | self.robot_truncateds
        done_all = bool(robot_done.all().item())
        
        # 是否所有机器人都是“自然终止”（到达或碰撞），没有超时
        all_naturally_terminated = bool(self.robot_terminateds.all().item())

        if done_all:
            if all_naturally_terminated:
                # 所有机器人都自然终止 → 终局
                terminateds["__all__"] = True
                truncateds["__all__"] = False
            else:
                # 至少有一个机器人是 timeout → 视为 time-limit 截断
                terminateds["__all__"] = False
                truncateds["__all__"] = True
        else:
            terminateds["__all__"] = False
            truncateds["__all__"] = False

        return obs, rewards, terminateds, truncateds, infos
    
    def _prepare_obs(self, id: int):
        """
        返回当前机器人 id 的观测:
        obs = [ self_obs(self.neighbor_obs_dim),
                neighbor_obs(num_use_neighbor_obs * self.neighbor_obs_dim),
                neighbor_valid_num(1),
                sdf_flat(map_size*map_size) ]
        其中:
          - self_obs = [x, y, goal_x, goal_y]（绝对坐标）
          - neighbor_obs 每个邻居 = [dx, dy, dgoal_x, dgoal_y]
            都是相对于当前机器人 (x,y) 的相对坐标
        """
        robot = self.robot_list[id]
        # ===== 1) self_obs: [x, y, goal_x, goal_y] (绝对坐标) =====
        robot_pos = robot.state[:self.self_obs_dim//2].ravel().astype(np.float32) # (self.self_obs_dim//2, )
        robot_goal = robot.goal[:self.self_obs_dim//2].ravel().astype(np.float32) # (self.self_obs_dim//2, )
        # 绝对坐标 -> 以世界中心为原点，除以 world_scale，约在 [-0.5, 0.5] 左右
        robot_pos_norm  = (robot_pos  - self.world_center) / self.world_scale
        robot_goal_norm = (robot_goal - self.world_center) / self.world_scale

        self_obs = np.concatenate([robot_pos_norm, robot_goal_norm], axis=0)  # (self.self_obs_dim,)        
        # ===== 2) neighbor_obs: 以当前机器人为中心的邻居 [rel_x, rel_y, rel_goal_x, rel_goal_y] =====
        neighbor_feats = []
        for j, other in enumerate(self.robot_list):
            if j == id:
                continue
            other_pos = other.state[:2].ravel()
            other_goal = other.goal[:2].ravel()

            dist = np.linalg.norm(other_pos - robot_pos)
            rel_vec = other_pos - robot_pos
            # 在sdf范围内 
            if self.esdf_cfg.x_min <= rel_vec[0] <= self.esdf_cfg.x_max and \
                self.esdf_cfg.y_min <= rel_vec[1] <= self.esdf_cfg.y_max:
                # 这里用“相对于当前机器人位置”的相对坐标, 同时进行缩放
                rel_pos = other_pos - robot_pos
                rel_goal = other_goal - robot_pos
                rel_pos_norm = rel_pos / self.world_scale
                rel_goal_norm = rel_goal / self.world_scale
                feat = np.concatenate([rel_pos_norm, rel_goal_norm], axis=0)
                neighbor_feats.append((dist, feat))

        # 按距离排序
        neighbor_feats.sort(key=lambda x: x[0])

        K = self.num_use_neighbor_obs
        neighbor_obs = np.zeros((K, self.neighbor_obs_dim), dtype=np.float32)

        # 实际可用邻居数量（可能为 0）
        M = min(len(neighbor_feats), K)
        for k in range(M):
            neighbor_obs[k] = neighbor_feats[k][1].astype(np.float32)

        neighbor_valid_num = float(M)  # 注意: 0~K 都合法

        neighbor_obs_flat = neighbor_obs.reshape(-1)  # (K * neighbor_obs_dim,)

        # ===== 3) ESDF/SDF 观测 =====
        pointcloud = lidar_to_pointcloud(robot.lidar.get_scan())
        sdf_grid = self.esdf.generate_sdf_from_pointcloud(pointcloud)
        sdf_grid = np.clip(sdf_grid, 0.0, self.r_max) / self.r_max # sdf归一化到 [0, 1]
        sdf_flat = sdf_grid.flatten()  # (map_size*map_size,)

        # ===== 4) 拼接总观测 =====
        obs = np.concatenate(
            [
                self_obs,                       # self_obs_dim
                neighbor_obs_flat,              # K * neighbor_obs_dim
                np.array([neighbor_valid_num], dtype=np.float32),  # 1
                sdf_flat,                       # map_size*map_size
            ],
            axis=0,
        )
        return obs.astype(np.float32)

    def _is_terminated(self, id: int):
        robot = self.robot_list[id]
        is_collision = robot.collision
        
        is_arrive = False
        robot_pos = robot.state[:self.self_obs_dim//2].ravel()
        robot_goal = robot.goal[:self.self_obs_dim//2].ravel()
        dis = np.linalg.norm(robot_goal - robot_pos)

        if dis < self.goal_threshold:
            is_arrive = True

        if is_collision or is_arrive:
            self.robot_terminateds[id] = True
        
        return self.robot_terminateds[id].item(), is_collision, is_arrive

    def _get_min_peer_distance(self, id: int) -> float:
        """
        返回当前机器人与其他机器人之间的最小欧氏距离（米）。
        用于奖励中的避让 shaping。
        """
        robot_pos = self.robot_list[id].state[:2].ravel()
        min_dist = float("inf")
        for j, other in enumerate(self.robot_list):
            if j == id:
                continue
            other_pos = other.state[:2].ravel()
            dist = float(np.linalg.norm(other_pos - robot_pos))
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    def _compute_travel_time_and_prior_dir(self, pos: np.ndarray, goal: np.ndarray) -> float:
        """
        用 PC-Planner 估计从 pos 到 goal 的“时间” T(x, goal), 以及先验方向。
        - 若 self.ntfields_nn 为 None, 则退化为欧氏距离近似。
        coords 采用和 self_obs 一样的归一化：[x,y,gx,gy] → 以 world_center 为原点并除以 world_scale。
        """
        # 退化：纯欧氏距离近似, 并且要归一化
        if self.ntfields_nn is None:
            diff = (goal - pos) / self.world_scale
            d = float(np.linalg.norm(diff))
            if d < 1e-8:
                prior_dir = np.zeros_like(diff)
            else:
                prior_dir = diff / d
            return d, torch.from_numpy(prior_dir).unsqueeze(0).to(self.device)

        pos_norm  = (pos  - self.world_center) / self.world_scale
        goal_norm = (goal - self.world_center) / self.world_scale

        start = torch.from_numpy(pos_norm).float().unsqueeze(0).to(self.device)  # [1,2]
        goal_t = torch.from_numpy(goal_norm).float().unsqueeze(0).to(self.device)  # [1,2]
        coords = torch.cat([start, goal_t], dim=-1)  # [1,4] = [start, goal]

        # 计算 tau 和 travel time T
        with torch.no_grad():
            tau, _ = self.ntfields_nn.out(coords)  # tau: [1,1]
            D = goal_t - start                     # [1,2]
            T0 = torch.einsum('ij,ij->i', D, D)    # ||goal-start||^2
            TT = torch.sqrt(T0) / tau[:, 0]        # [1]

        # 用 NTFields 的梯度给一个先验方向（对 start 的梯度）
        grad_all = self.ntfields_nn.calculate_gradient(coords)  # [1,4] = [dT/dx_start(2), dT/dx_goal(2)]
        grad_start = grad_all[:, :2]                            # [1,2]
        grad_norm = torch.norm(grad_start, dim=-1, keepdim=True)
        prior_dir = grad_start / (grad_norm + 1e-8)             # [1,2]

        return float(TT.item()), prior_dir
    
    def _single_neighbor_potential(self, d: float) -> float:
        """
        连续三段势场 φ(d):

        区间划分:
        - d >= d_sense       : 0
        - d_safe <= d < d_sense   : 远距离轻微惩罚 (0 -> k_far)
        - d_danger <= d < d_safe  : 安全壳 (k_mid -> k_far)
        - 0 < d < d_danger        : 危险区 (k_mid -> k_max, 二次增强)
        - d <= 0                  : 极端情况, 直接给 k_max

        通过线性 / 二次插值保证在 d_danger、d_safe、d_sense 处函数值连续。
        """
        r = float(self.robot_radius)

        # 三个半径门槛
        d_danger = 2.0 * r + 0.3   # 危险壳
        d_safe   = d_danger + 0.5  # 安全壳外缘
        d_sense  = d_safe   + 1.0  # 感知上限

        # 三段对应的大致势场强度 (可调)
        k_far  = 0.3   # 远区
        k_mid  = 1.0   # 安全壳
        k_max  = 3.0   # 危险区极大值

        d = float(d)

        # 远到完全不关心
        if d >= d_sense:
            return 0.0

        # 远区: [d_safe, d_sense]，从 0 → k_far，线性插值
        if d >= d_safe:
            t = (d_sense - d) / max(d_sense - d_safe, 1e-6)  # t ∈ (0,1]
            phi = k_far * t        # φ(d_sense)=0, φ(d_safe)=k_far
            return float(phi)

        # 安全壳: [d_danger, d_safe)，从 k_mid → k_far，线性插值
        if d >= d_danger:
            s = (d - d_danger) / max(d_safe - d_danger, 1e-6)  # s ∈ [0,1)
            phi = k_mid + (k_far - k_mid) * s                  # φ(d_danger)=k_mid, φ(d_safe^-)=k_far
            return float(phi)

        # 危险区: (0, d_danger)，从 k_mid → k_max，二次增长
        if d > 0.0:
            u = (d_danger - d) / max(d_danger - 2.0 * r, 1e-6)  # u ∈ (0,1]
            phi = k_mid + (k_max - k_mid) * (u * u)   # φ(d_danger)=k_mid, φ(0)≈k_max
            return float(phi)

        # 理论上 d=0 几乎不会发生, 给最大势场
        return k_max

    def _compute_peer_potential(self, id: int):
        """
        聚合当前机器人与所有邻居的势场值:
        - 返回 (Phi, min_peer_dist, in_peer_danger)
        - Phi 用 smooth max (log-mean-exp) 聚合各个 φ(d_j)，既关注最近危险邻居，又保证连续性。
        """
        robot_pos = self.robot_list[id].state[:2].ravel()
        r = float(self.robot_radius)

        d_danger = 2.0 * r + 0.3
        d_safe   = d_danger + 0.5

        phi_list = []
        min_dist = float("inf")
        in_peer_danger = False

        for j, other in enumerate(self.robot_list):
            if j == id:
                continue
            other_pos = other.state[:2].ravel()
            d = float(np.linalg.norm(other_pos - robot_pos))
            if d < min_dist:
                min_dist = d

            phi_j = self._single_neighbor_potential(d)
            if phi_j > 0.0:
                phi_list.append(phi_j)

            # 只要有邻居进入 d_safe 内，就认为在“避让区”
            if d < d_safe:
                in_peer_danger = True

        # 没有邻居或者都太远
        if not phi_list:
            return 0.0, min_dist, False

        arr = np.asarray(phi_list, dtype=np.float32)

        # smooth max: Phi ≈ max φ_j
        beta = 5.0
        m = float(arr.max())
        Phi = m + (1.0 / beta) * float(np.log(np.mean(np.exp(beta * (arr - m)))))

        return float(Phi), min_dist, in_peer_danger

    def _calculate_reward(
        self,
        id: int,
        action,
        obs: np.ndarray,
        is_collision: bool,
        is_arrive: bool,
        is_truncated: bool,
    ):
        robot = self.robot_list[id]

        # ---------------- 权重 ----------------
        w_T_progress   = 6.0    # Time Field 进度
        w_step         = 0.01   # 每步时间成本

        # 势场相关：变化项 + 静态项
        w_peer_delta   = 2.0    # 势场变化惩罚 (ΔΦ)
        w_peer_static  = 0.3    # 高势场静态惩罚 Φ（只在危险区内）

        w_obst_delta  = 1.5     # 惩罚“更靠近”
        w_obst_static = 0.05    # 贴墙静态惩罚

        w_speed_base   = 0.005  # 速度惩罚
        w_speed_danger = 0.08   # 危险区域的速度惩罚加成

        w_prior_align  = 0.05   # 先验方向对齐

        bonus_arrive          = 25.0
        pen_timeout           = 10.0
        pen_collision_robot   = 35.0
        pen_collision_obstacle= 25.0

        r = float(self.robot_radius)

        # 三个半径门槛
        d_danger = 2.0 * r + 0.3   # 危险壳
        d_safe   = d_danger + 0.5  # 安全壳外缘
        d_sense  = d_safe   + 1.0  # 感知上限

        # ---------------- 当前几何量 ----------------
        robot_pos  = robot.state[:2].ravel()
        robot_goal = robot.goal[:2].ravel()

        cur_action = np.asarray(action, dtype=np.float32)
        speed = float(np.linalg.norm(cur_action))

        # 1) NTFields 时间进度 ΔT + 先验方向
        T_curr, prior_dir = self._compute_travel_time_and_prior_dir(robot_pos, robot_goal)
        T_prev = float(self.robot_last_T[id].item())
        delta_T = float(np.clip(T_prev - T_curr, -1.0, 1.0))

        # 2) 多机器人势场 Φ 及其变化（不归一化）
        Phi_curr, min_peer_dist, in_peer_danger = self._compute_peer_potential(id)
        Phi_prev = float(self.robot_last_peer_phi[id].item())
        delta_Phi = Phi_curr - Phi_prev      # >0 表示变更危险

        # 3) 障碍物接近程度
        scan = robot.lidar.get_scan()
        ranges = np.asarray(scan["ranges"], dtype=np.float32)
        valid = np.isfinite(ranges)
        if valid.any():
            min_lidar = float(np.clip(ranges[valid].min(), 0.0, self.r_max))
        else:
            min_lidar = self.r_max

        clear_safe = 2.0 * self.robot_radius + 0.2
        in_obst_danger = (min_lidar < clear_safe)

        if in_obst_danger:
            loss_clear = (clear_safe - min_lidar) / clear_safe
            loss_clear = float(np.clip(loss_clear, 0.0, 1.0))
        else:
            loss_clear = 0.0
        
        loss_prev  = float(self.robot_last_obst_loss[id].item())
        delta_loss = loss_clear - loss_prev  # >0 更靠近障碍
        self.robot_last_obst_loss[id] = float(loss_clear)
        # ---------------- 各项 reward 组合 ----------------
        reward = 0.0

        # 安全区内的 T 进度奖励, 但是避免一开始避让就回头
        # 0~1，越近越小
        peer_gate = float(np.clip((min_peer_dist - d_danger) / (d_sense - d_danger), 0.0, 1.0))
        obst_gate = 0.0 if in_obst_danger else 1.0
        r_T_progress = w_T_progress * peer_gate * obst_gate * delta_T

        reward += r_T_progress

        # 每步时间成本
        r_step = -w_step
        reward += r_step

        # 势场变化惩罚：ΔΦ>0 → 靠近别人 → 惩罚
        # r_peer_delta = -w_peer_delta * delta_Phi
        w_peer_away = 0.3 * w_peer_delta  # 小于惩罚权重
        r_peer_delta = -w_peer_delta * max(delta_Phi, 0.0) + w_peer_away * max(-delta_Phi, 0.0)
        # 高势场静态惩罚：只有在“避让区”才启用，否则远处偶尔靠近一下惩罚交给 ΔΦ
        r_peer_static = -w_peer_static * Phi_curr if in_peer_danger else 0.0

        reward += r_peer_delta + r_peer_static

        # 障碍物清距惩罚
        if in_obst_danger:
            r_obst_delta  = -w_obst_delta  * max(delta_loss, 0.0)
            r_obst_static = -w_obst_static * loss_clear
        else:
            r_obst_delta  = 0.0
            r_obst_static = 0.0
        
        reward += r_obst_delta + r_obst_static

        # 速度惩罚
        # 风险度：0~1
        peer_risk = 0.0
        if in_peer_danger:
            peer_risk = float(np.clip((d_safe - min_peer_dist) / (d_safe - d_danger), 0.0, 1.0))

        r_speed = -(w_speed_base + w_speed_danger * peer_risk) * (speed ** 2)
        reward += r_speed

        # 安全区内的先验方向对齐奖励
        r_prior_align = 0.0
        align_gate = peer_gate * obst_gate
        if speed > 1e-4:
            cur_dir = cur_action / (speed + 1e-8)
            prior_np = prior_dir.detach().cpu().numpy().reshape(-1)
            pn = float(np.linalg.norm(prior_np))
            if pn > 1e-6:
                prior_unit = prior_np / pn
                cos_theta = float(np.dot(cur_dir, prior_unit))
                cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
                r_prior_align = w_prior_align * cos_theta
                reward += r_prior_align * align_gate

        # 终止项
        r_arrive   = bonus_arrive if is_arrive    else 0.0
        r_timeout  = -pen_timeout if is_truncated else 0.0

        if is_collision:
            if min_peer_dist < 2.0 * self.robot_radius + 0.05:
                r_coll_robot   = -pen_collision_robot
                r_coll_obst    = 0.0
            else:
                r_coll_robot   = 0.0
                r_coll_obst    = -pen_collision_obstacle
        else:
            r_coll_robot = 0.0
            r_coll_obst  = 0.0

        reward += r_arrive + r_timeout + r_coll_robot + r_coll_obst

        # 更新缓存
        self.robot_last_T[id]        = float(T_curr)
        self.robot_last_peer_phi[id] = float(Phi_curr)
        self.robot_last_action[id]   = cur_action

        # 如果你已经在 step 里把 info 接出来，可以继续返回 info；否则只返回 reward。
        info = {
            "r_total":        float(reward),
            "r_T_progress":   float(r_T_progress),
            "r_step":         float(r_step),
            "r_peer_delta":   float(r_peer_delta),
            "r_peer_static":  float(r_peer_static),
            "r_obst_delta":   float(r_obst_delta),
            "r_obst_static":  float(r_obst_static),
            "r_speed":        float(r_speed),
            "r_prior_align_raw":  float(r_prior_align),
            "r_prior_align":  float(r_prior_align * align_gate),
            "r_arrive":       float(r_arrive),
            "r_timeout":      float(r_timeout),
            "r_coll_robot":   float(r_coll_robot),
            "r_coll_obst":    float(r_coll_obst),

            "T_curr":         float(T_curr),
            "delta_T":        float(delta_T),
            "Phi_curr":       float(Phi_curr),
            "delta_Phi":      float(delta_Phi),
            "min_peer_dist":  float(min_peer_dist),
            "min_lidar":      float(min_lidar),
            "in_peer_danger": bool(in_peer_danger),
            "in_obst_danger": bool(in_obst_danger),
        }

        return float(reward), info

    def reset(self, robot_states=None, robot_goals=None):
        if robot_states is not None:
            assert len(robot_states) == self.num_robots
        if robot_goals is not None:
            assert len(robot_goals) == self.num_robots
        
        if robot_states is not None and robot_goals is not None:
            for i, state in enumerate(robot_states):
                self.robot_list[i].set_state(state=np.array(state), init=True)

            for i, goal in enumerate(robot_goals):
                self.robot_list[i].set_goal(np.array(goal), init=True)
        else:
            for robot in self.robot_list:
                self.reset_robot(robot=robot)
        
        # 重置环境
        self.env.reset()
        self.robot_times = torch.zeros(self.num_robots, dtype=torch.int32)
        self.robot_terminateds = torch.zeros(self.num_robots, dtype=torch.bool)
        self.robot_truncateds = torch.zeros(self.num_robots, dtype=torch.bool)
        self.robot_last_action = np.zeros((self.num_robots, 2), dtype=np.float32)
        self.robot_last_obst_loss = torch.zeros(self.num_robots, dtype=torch.float32)

        positions = []
        goals = []
        for robot in self.robot_list:
            positions.append(robot.state[:2].ravel())
            goals.append(robot.goal[:2].ravel())

        positions = np.asarray(positions, dtype=np.float32)
        goals = np.asarray(goals, dtype=np.float32)

        # 初始化每个机器人的初始 goal 距离
        for i in range(self.num_robots):
            dis = np.linalg.norm(goals[i] - positions[i])
            self.robot_last_goal_dist[i] = float(dis)

        # 初始化Time Field时间缓存 T_i(0) 和势场 Φ_i(0)
        for i in range(self.num_robots):
            T0, _ = self._compute_travel_time_and_prior_dir(positions[i], goals[i])
            self.robot_last_T[i] = float(T0)
            Phi0, _, _ = self._compute_peer_potential(i)
            self.robot_last_peer_phi[i] = float(Phi0)

        obs = {}
        infos = {}
        for id in range(self.num_robots):
            robot_id = f'robot_{id}'
            obs[robot_id] = self._prepare_obs(id)
            infos[robot_id] = {}

        return obs, infos

    def reset_robot(
        self,
        robot,
        robot_state=None,
        robot_goal=None
    ):
        if robot_state is not None:
            robot.set_state(np.array(robot_state), init=True)
        else:
            free_pos = robot.get_random_free_position(
                self.env.objects,
                free = True,
                range_limits=[
                    [self.x_range[0] + 1, self.y_range[0] + 1, 0],
                    [self.x_range[1] - 1, self.y_range[1] - 1, 0],
                ],
            )
            robot.set_state(free_pos, init=True)
            
        if robot_goal is not None:
            robot.set_goal(np.array(robot_goal), init=True)
        else:
            free_pos = robot.get_random_free_position(
                self.env.objects,
                free = True,
                filter_goal = True,
                range_limits=[
                    [self.x_range[0] + 1, self.y_range[0] + 1, 0],
                    [self.x_range[1] - 1, self.y_range[1] - 1, 0],
                ],
            )
            robot.set_goal(free_pos, init=True)
        # 重置碰撞等信息
        robot.reset()