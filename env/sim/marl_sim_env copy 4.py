import irsim
import numpy as np
import torch

from model.models import NN

from .esdf_2d_simple import ESDF_2d
from ..params import SceneConfig
from utils.utils import lidar_to_pointcloud
from .sim_env import SIM_ENV

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

            # 计算reward（传入当前动作 + obs）
            rewards[robot_id] = self._calculate_reward(
                id,
                raw_actions[id],
                obs[robot_id],
                is_collision,
                is_arrive,
                is_truncated,
            )
        
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
    
    def _compute_travel_time(self, pos: np.ndarray, goal: np.ndarray) -> float:
        """
        用 PC-Planner 估计从 pos 到 goal 的“时间” T(x, goal)。
        - 若 self.ntfields_nn 为 None, 则退化为欧氏距离近似。
        coords 采用和 self_obs 一样的归一化：[x,y,gx,gy] → 以 world_center 为原点并除以 world_scale。
        """
        # 退化：纯欧氏距离近似, 并且要归一化
        if self.ntfields_nn is None:
            d = float(np.linalg.norm((goal - pos) / self.world_scale))
            return d

        pos_norm  = (pos  - self.world_center) / self.world_scale
        goal_norm = (goal - self.world_center) / self.world_scale
        with torch.no_grad():
            start = torch.from_numpy(pos_norm).unsqueeze(0).to(self.device)  # [1,2]
            goal = torch.from_numpy(goal_norm).unsqueeze(0).to(self.device)  # [1,2]

            coords = torch.cat([start, goal], dim=-1)  # [1,4]
            tau, _ = self.ntfields_nn.out(coords)  # tau: [1,1]

            D = goal - start
            T0 = torch.einsum('ij,ij->i', D, D)
            TT = torch.sqrt(T0)/tau[:, 0]  # [1]

        return TT.detach().item()

    def _calculate_reward(
        self,
        id: int,
        action,
        obs: np.ndarray,
        is_collision: bool,
        is_arrive: bool,
        is_truncated: bool,
    ):
        """
        奖励设计（适中复杂度）：
        1) 目标距离进度
        2) 时间成本
        3) 机器人-机器人距离 shaping + 简单方向避让
        4) 障碍清距惩罚
        5) 速度 / 平滑惩罚（简单版）
        6) 方向对齐奖励
        7) 终止奖励 / 惩罚
        """
        robot = self.robot_list[id]

        # ---------------- 权重（建议先这样） ----------------
        w_progress = 2.0      # 接近目标
        w_time     = 0.01     # 每步时间成本

        # 机器人-机器人相关
        w_peer_mid   = 1.5    # 中等距离惩罚
        w_peer_near  = 4.0    # 进入“安全壳”内的强惩罚
        w_peer_dir   = 2.0    # 朝向别的机器人运动的惩罚

        # 障碍物相关
        w_clear      = 3.0    # 靠近障碍物惩罚

        # 速度 / 平滑
        w_speed      = 0.02   # 速度惩罚（整体偏小）
        w_smooth_safe = 0.05  # 安全时平滑
        w_smooth_danger = 0.01# 危险时弱平滑

        # 对齐目标方向
        w_align      = 0.2

        # 终止项
        bonus_arrive          = 25.0
        pen_timeout           = 8.0
        pen_collision_robot   = 35.0
        pen_collision_obstacle= 20.0

        # ---------- 1) 目标进度 ----------
        robot_pos  = robot.state[:2].ravel()
        robot_goal = robot.goal[:2].ravel()
        dist = float(np.linalg.norm(robot_goal - robot_pos))

        last_dist = float(self.robot_last_goal_dist[id].item())
        progress = last_dist - dist           # >0 表示靠近目标
        progress = float(np.clip(progress, -0.5, 0.5))
        self.robot_last_goal_dist[id] = dist

        reward = 0.0
        reward += w_progress * progress
        reward -= w_time

        # ---------- 2) 最近机器人距离 + 方向 ----------
        # 找最近机器人及其位置
        min_peer_dist = float("inf")
        nearest_pos = None
        for j, other in enumerate(self.robot_list):
            if j == id:
                continue
            other_pos = other.state[:2].ravel()
            d = float(np.linalg.norm(other_pos - robot_pos))
            if d < min_peer_dist:
                min_peer_dist = d
                nearest_pos = other_pos

        # 定义三个区间：
        #   d < d_hard        ：几乎要碰撞了，强惩罚
        #   d_hard ~ d_soft   ：安全壳内，中等惩罚
        #   d_soft ~ d_far    ：略近，给一点轻微 shaping
        if nearest_pos is not None:
            r = self.robot_radius
            d_hard = 2.0 * r + 0.2   # 硬核安全壳
            d_soft = d_hard + 0.5    # 软壳
            d_far  = d_soft + 0.6    # 远距离 shaping 上限

            in_peer_danger = False

            if min_peer_dist < d_hard:
                # 壳内：强惩罚，随距离二次增长；上限 clip 一下
                depth = max(d_hard - min_peer_dist, 0.0)
                depth_norm = float(np.clip(depth / d_hard, 0.0, 1.0))
                peer_pen = w_peer_near * (1.0 + 2.0 * (depth_norm ** 2))
                peer_pen = min(peer_pen, 3.0 * w_peer_near)  # clip
                reward -= peer_pen
                in_peer_danger = True

            elif min_peer_dist < d_soft:
                # 中间壳：线性惩罚
                t = (d_soft - min_peer_dist) / max(d_soft - d_hard, 1e-6)  # 0~1
                peer_pen = w_peer_mid * t
                reward -= peer_pen
                in_peer_danger = True

            elif min_peer_dist < d_far:
                # 稍微近一点：轻微惩罚，避免“刚好贴边走”
                t = (d_far - min_peer_dist) / max(d_far - d_soft, 1e-6)  # 0~1
                peer_pen = 0.5 * w_peer_mid * t
                reward -= peer_pen
                # 这里不认为是 danger，主要是很轻微的 shaping
                in_peer_danger = False

            # ---------- 简单方向避让：只在 danger 区 ----------
            cur_action = np.asarray(action, dtype=np.float32)
            speed = float(np.linalg.norm(cur_action))
            if in_peer_danger and speed > 1e-6:
                dir_vec = nearest_pos - robot_pos
                dir_norm = float(np.linalg.norm(dir_vec))
                if dir_norm > 1e-6:
                    u = dir_vec / dir_norm       # 指向最近机器人的方向
                    v = cur_action / speed       # 当前动作方向
                    cos_theta = float(np.dot(v, u))
                    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))

                    # 只惩罚“往别人那边走”的分量
                    if cos_theta > 0.0:
                        # 让越靠近 d_hard，惩罚系数越大
                        if min_peer_dist < d_soft:
                            alpha = (d_soft - min_peer_dist) / max(d_soft - d_hard, 1e-6)
                        else:
                            alpha = 0.0
                        alpha = float(np.clip(alpha, 0.0, 1.0))
                        reward -= w_peer_dir * alpha * (cos_theta ** 2)
        else:
            in_peer_danger = False
            cur_action = np.asarray(action, dtype=np.float32)
            speed = float(np.linalg.norm(cur_action))

        # ---------- 3) 障碍清距惩罚 ----------
        scan = robot.lidar.get_scan()
        ranges = np.asarray(scan["ranges"], dtype=np.float32)
        valid = np.isfinite(ranges)
        if valid.any():
            min_lidar = float(np.clip(ranges[valid].min(), 0.0, self.r_max))
        else:
            min_lidar = self.r_max

        clear_safe = max(2.0 * self.robot_radius + 0.1, 0.4)
        in_obst_danger = (min_lidar < clear_safe)

        if in_obst_danger:
            # 线性惩罚到 clear_safe，其实已经够用了
            loss_clear = (clear_safe - min_lidar) / clear_safe
            loss_clear = float(np.clip(loss_clear, 0.0, 1.5))
            reward -= w_clear * loss_clear

        # ---------- 4) 速度 + 简单平滑 ----------
        prev_action = self.robot_last_action[id]

        # 速度惩罚：全局都在，但权重很小
        reward -= w_speed * (speed ** 2)

        # 动作变化惩罚：只看一阶差分，简单稳定
        delta_a = float(np.linalg.norm(cur_action - prev_action))
        if (not in_peer_danger) and (not in_obst_danger):
            w_smooth = w_smooth_safe
        else:
            w_smooth = w_smooth_danger
        reward -= w_smooth * (delta_a ** 2)

        self.robot_last_action[id] = cur_action

        # ---------- 5) 目标方向对齐（只在不需要紧急避让时） ----------
        if (not in_peer_danger) and (not in_obst_danger) and speed > 1e-6:
            goal_vec = robot_goal - robot_pos
            goal_norm = float(np.linalg.norm(goal_vec))
            if goal_norm > 1e-6:
                cos_align = float(np.dot(cur_action, goal_vec) / (speed * goal_norm))
                cos_align = float(np.clip(cos_align, -1.0, 1.0))
                reward += w_align * cos_align

        # ---------- 6) 终止项 ----------
        if is_arrive:
            reward += bonus_arrive

        if is_truncated:
            reward -= pen_timeout

        if is_collision:
            # 已经撞上去：惩罚要明显大于所有危险区内的 step 级惩罚
            if nearest_pos is not None and min_peer_dist < d_soft:
                reward -= pen_collision_robot
            else:
                reward -= pen_collision_obstacle

        return float(reward)


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

        # 初始化Time Field时间缓存 T_i(0)
        for i in range(self.num_robots):
            T0 = self._compute_travel_time(positions[i], goals[i])
            self.robot_last_T[i] = float(T0)
            print(T0)

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