import irsim
import numpy as np
import torch

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
        self.robot_ema_action  = np.zeros((self.num_robots, 2), dtype=np.float32)   

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
        奖励设计（复杂版：提前预判 + 多步平滑），经过重新标定：
        1) 目标距离进度
        2) 时间成本
        3) 机器人-机器人间距 + 方向避让（提前预判）
        4) 障碍清距惩罚
        5) 距离相关的速度惩罚（但整体较温和）
        6) 多步方向平滑（EMA）
        7) 终止奖励/惩罚
        """
        robot = self.robot_list[id]

        # ---------------- 权重（调小了整体惩罚尺度） ----------------
        w_progress = 2.0        # 主正向信号，保留
        w_time = 0.01           # 轻时间惩罚

        # 机器人之间：距离 + 方向
        w_peer_dist = 1.0       # 原来 2.5，显著调低
        w_peer_dir  = 1.5       # 原来 3.0，也调低一半

        # 障碍物距离
        w_clear = 2.0           # 原来 4.0，减半

        # 速度 / 平滑
        base_w_speed = 0.005    # 原来 0.01，降一半
        w_smooth_far  = 0.04    # 安全区平滑权重（你前面说的 0.04）
        w_smooth_near = 0.01    # 危险区平滑很弱，只做一点点抑制

        # 目标方向对齐
        w_align = 0.4           # 稍微弱一点，避免 push 太狠

        # 终止项
        bonus_arrive = 15.0     # 从 20 调低一点
        pen_timeout = 5.0
        pen_collision_robot = 20.0   # 从 30 降低，但仍 > 近距离惩罚
        pen_collision_obstacle = 12.0

        # ---------- 1) 目标进度 ----------
        robot_pos = robot.state[:2].ravel()
        robot_goal = robot.goal[:2].ravel()
        dist = float(np.linalg.norm(robot_goal - robot_pos))

        last_dist = float(self.robot_last_goal_dist[id].item())
        progress = last_dist - dist                      # >0 表示靠近目标
        progress = float(np.clip(progress, -0.5, 0.5))   # 单步进度限制在 [-0.5,0.5]
        self.robot_last_goal_dist[id] = dist

        reward = 0.0
        reward += w_progress * progress                  # 单步大概在 [-1,1]
        reward -= w_time

        # ---------- 2) 最近机器人 + 危险判定 ----------
        neighbors = []
        for j, other in enumerate(self.robot_list):
            if j == id:
                continue
            other_pos = other.state[:2].ravel()
            d = float(np.linalg.norm(other_pos - robot_pos))
            neighbors.append((d, other_pos))

        min_peer_dist = float("inf")
        nearest_pos = None
        for d, pos_j in neighbors:
            if d < min_peer_dist:
                min_peer_dist = d
                nearest_pos = pos_j

        peer_safe = 2.0 * self.robot_radius + 0.5
        peer_warn = peer_safe + 0.5

        in_peer_danger = (nearest_pos is not None) and (min_peer_dist < peer_warn)

        # 2.1 距离 barrier（外壳：soft，壳内：稍大）
        if in_peer_danger:
            eps = 1e-3
            if min_peer_dist > peer_safe:
                # 壳外 [peer_safe, peer_warn]：用 1/x 但做上界限制，防止爆炸
                x = min_peer_dist - peer_safe
                proximity = 1.0 / max(x, eps) - 1.0 / (peer_warn - peer_safe + eps)
                # 限制 barrier 的最大值，避免极端步数上 reward 爆炸
                proximity = float(np.clip(proximity, 0.0, 3.0))
                reward -= w_peer_dist * proximity
            else:
                # 壳内：越靠近中心惩罚越大，但也做上界
                depth = peer_safe - min_peer_dist
                depth_norm = float(np.clip(depth / peer_safe, 0.0, 1.0))
                inside_penalty = (1.0 + 2.0 * (depth_norm ** 2))  # 原来 1+4x^2，改成 1+2x^2
                inside_penalty = float(np.clip(inside_penalty, 1.0, 3.0))
                reward -= w_peer_dist * inside_penalty

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
            # 线性惩罚，尺度较温和
            penalty_clear = (clear_safe - min_lidar) / clear_safe  # [0,1]
            reward -= w_clear * penalty_clear

        # ---------- 4) 速度 + EMA 平滑 ----------
        cur_action = np.asarray(action, dtype=np.float32)
        prev_action = self.robot_last_action[id]
        ema_prev = self.robot_ema_action[id]

        speed = float(np.linalg.norm(cur_action))

        # 4.1 距离相关速度惩罚：放大系数也限制住
        peer_factor = 0.0
        if in_peer_danger and min_peer_dist > 0.0:
            peer_factor = (peer_warn - min_peer_dist) / (peer_warn - peer_safe + 1e-6)
            peer_factor = float(np.clip(peer_factor, 0.0, 1.0))

        obst_factor = 0.0
        if min_lidar < clear_safe * 1.5:
            obst_factor = (clear_safe * 1.5 - min_lidar) / (clear_safe * 1.5)
            obst_factor = float(np.clip(obst_factor, 0.0, 1.0))

        # 放大最大到 1+2 = 3（之前是 1+4 = 5）
        factor = 1.0 + 2.0 * max(peer_factor, obst_factor)
        w_speed_eff = base_w_speed * factor
        reward -= w_speed_eff * (speed ** 2)

        # 4.2 多步 EMA 平滑
        alpha = 0.7  # 历史记忆长度
        ema_new = alpha * ema_prev + (1.0 - alpha) * cur_action
        self.robot_ema_action[id] = ema_new

        ema_speed_prev = float(np.linalg.norm(ema_prev))
        dir_change = 0.0
        if speed > 1e-6 and ema_speed_prev > 1e-6:
            v_now = cur_action / speed
            v_hist = ema_prev / ema_speed_prev
            cos_dir = float(np.dot(v_now, v_hist))
            cos_dir = float(np.clip(cos_dir, -1.0, 1.0))
            # dir_change ∈ [0,2]，0 同向、2 反向
            dir_change = 1.0 - cos_dir

        if (not in_peer_danger) and (not in_obst_danger):
            w_smooth = w_smooth_far
        else:
            w_smooth = w_smooth_near

        # dir_change^2 ∈ [0,4]，乘上 w_smooth，单步最多 0.16 左右
        reward -= w_smooth * (dir_change ** 2)

        # 更新上一时刻动作
        self.robot_last_action[id] = cur_action

        # ---------- 5) 机器人方向避让（最近一个） ----------
        if in_peer_danger and speed > 1e-6 and nearest_pos is not None:
            dir_vec = nearest_pos - robot_pos
            dir_norm = float(np.linalg.norm(dir_vec))
            if dir_norm > 1e-6:
                u = dir_vec / dir_norm           # 指向最近机器人的方向
                v = cur_action / speed           # 当前动作方向
                cos_theta = float(np.dot(v, u))
                cos_theta = float(np.clip(cos_theta, -1.0, 1.0))

                alpha_dir = (peer_warn - min_peer_dist) / (peer_warn - peer_safe + 1e-6)
                alpha_dir = float(np.clip(alpha_dir, 0.0, 1.0))

                if cos_theta > 0:
                    # 冲向别人：惩罚，做一个上界，防止叠加爆炸
                    penalty_dir = (cos_theta ** 2)  # [0,1]
                    reward -= w_peer_dir * alpha_dir * penalty_dir
                else:
                    # 朝反方向离开：给小奖励
                    bonus_dir = ((-cos_theta) ** 2)  # [0,1]
                    reward += 0.3 * w_peer_dir * alpha_dir * bonus_dir

        # ---------- 6) 对齐目标方向奖励（只有在机器人和障碍都不危险时） ----------
        if (not in_peer_danger) and (not in_obst_danger) and speed > 1e-6:
            goal_vec = robot_goal - robot_pos
            goal_norm = float(np.linalg.norm(goal_vec))
            if goal_norm > 1e-6:
                cos_align = float(np.dot(cur_action, goal_vec) / (speed * goal_norm))
                cos_align = float(np.clip(cos_align, -1.0, 1.0))
                reward += w_align * cos_align

        # ---------- 7) 终止项 ----------
        if is_arrive:
            reward += bonus_arrive

        if is_truncated:
            reward -= pen_timeout

        if is_collision:
            # 碰撞惩罚要明显 > 一般危险区的单步惩罚
            if in_peer_danger and (min_peer_dist < peer_safe):
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
        self.robot_ema_action  = np.zeros((self.num_robots, 2), dtype=np.float32)

        # 初始化每个机器人的初始 goal 距离
        for i, robot in enumerate(self.robot_list):
            robot_pos = robot.state[:2].ravel()
            robot_goal = robot.goal[:2].ravel()
            dis = np.linalg.norm(robot_goal - robot_pos)
            self.robot_last_goal_dist[i] = float(dis)

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