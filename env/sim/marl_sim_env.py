import irsim
import math
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
        self.render = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=self.render
        )
        self.num_robots = len(self.env.robot_list)
        self.robot_list = self.env.robot_list
        self.obstacle_list = self.env.obstacle_list
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
        self.v_max = scene_cfg.robot_params.v_max

        # reward 配置
        self.reward_cfg = scene_cfg.reward_params
        self.robot_times = torch.zeros(self.num_robots, dtype=torch.int32)  # time step counter
        self.robot_terminateds = torch.zeros(self.num_robots, dtype=torch.bool) # terminate
        self.robot_truncateds = torch.zeros(self.num_robots, dtype=torch.bool) # truncate

        self.robot_last_goal_dist = torch.zeros(self.num_robots, dtype=torch.float32)
        self.robot_radius = scene_cfg.robot_params.radius

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
        self.robot_last_prior_dir = np.zeros((self.num_robots, 2), dtype=np.float32)
        self.robot_last_peer_phi = torch.zeros(self.num_robots, dtype=torch.float32)
        self.robot_last_obst_loss = torch.zeros(self.num_robots, dtype=torch.float32)

        self._env_step_id = 0
    # only alive robots have action
    def step(self, action_dict): # action_dict: dict of(robot_id, [alpha, theta]), [-1, 1]
        applied_actions = []
        action_ids = []
        raw_actions = {}   # 归一化动作 (alpha/theta)
        exec_dirs = {}     # 实际执行方向（单位向量）
        exec_speeds = {}   # 实际执行速度（m/s）
        # -------- 1) 用“上一次 obs 的 prior_dir”把 (alpha, theta) -> (dx, dy) 并执行 --------
        for robot_id, action in action_dict.items():
            id = int(robot_id.split('_')[-1]) # 'robot_0' -> 0
            action_ids.append(id)
            
            alpha_raw = float(action[0])
            theta_raw = float(action[1])
            raw_actions[id] = np.array(action, dtype=np.float32)

            # alpha: [-1,1] -> [0,1]
            alpha01 = (alpha_raw + 1.0) / 2.0
            alpha01 = float(np.clip(alpha01, 0.0, 1.0))

            # theta: [-1,1] -> [-pi, pi]
            theta = float(theta_raw) * math.pi

            # prior_dir 已经是单位向量了
            prior_prev = self.robot_last_prior_dir[id].astype(np.float32)

            # 旋转 prior
            c, s = math.cos(theta), math.sin(theta)
            dir_x = c * float(prior_prev[0]) - s * float(prior_prev[1])
            dir_y = s * float(prior_prev[0]) + c * float(prior_prev[1])
            exec_dir = np.array([dir_x, dir_y], dtype=np.float32)

            v = alpha01 * float(self.v_max)  # 类似于m/s
            dx = float(exec_dir[0]) * v * float(self.move_single_step)
            dy = float(exec_dir[1]) * v * float(self.move_single_step)

            applied_actions.append([dx, dy])
            exec_dirs[id] = exec_dir
            exec_speeds[id] = v

        self.env.step(action=applied_actions, action_id=action_ids)
        if self.render:
            self.env.render()
        self._env_step_id += 1

        obs = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        # -------- 2) 先算 reward（内部算一次 T/prior 并缓存），再拼 obs --------
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
            
            r, info = self._calculate_reward(
                id=id,
                action_raw=raw_actions[id],
                prior_dir_prev=self.robot_last_prior_dir[id].copy(),  # copy 防止未来改动引入引用问题
                exec_dir=exec_dirs[id],
                exec_speed=exec_speeds[id],
                is_collision=is_collision,
                is_arrive=is_arrive,
                is_truncated=is_truncated,
            )
            rewards[robot_id] = r
            infos[robot_id] = info
            infos[robot_id]["_env_step_id"] = self._env_step_id
            
            # reward 已经更新了 prior_dir_curr 缓存，这里直接用缓存拼 obs
            obs[robot_id] = self._prepare_obs(id)
        
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
                sdf_flat(map_size*map_size),
                prior_dir(nt_dim) ]
        其中:
          - self_obs = [x, y, goal_x, goal_y]（绝对坐标）
          - neighbor_obs 每个邻居 = [dx, dy, dgoal_x, dgoal_y]
            都是相对于当前机器人 (x,y) 的相对坐标
          - prior_dir 为 env 预先计算好的单位方向（与 policy 的 theta 旋转基准一致）
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

        # ===== 4) prior action (direction) =====
        prior_dir = self.robot_last_prior_dir[id].astype(np.float32)  # (nt_dim,)

        # ===== 5) 拼接总观测 =====
        obs = np.concatenate(
            [
                self_obs,                       # self_obs_dim
                neighbor_obs_flat,              # K * neighbor_obs_dim
                np.array([neighbor_valid_num], dtype=np.float32),  # 1
                sdf_flat,                       # map_size*map_size
                prior_dir,                      # nt_dim
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
    
    @staticmethod
    def _dist_point_to_obb_boundary(p: np.ndarray, c: np.ndarray,
                                   half_l: float, half_w: float, yaw: float) -> float:
        """
        计算“点 p 到旋转矩形(OBB)外边界”的最短距离（单位：米）。

        说明：
        - 这里的“点”指机器人中心点 p。
        - 这里的“矩形”是障碍物矩形：中心 c，半长 half_l，半宽 half_w，旋转角 yaw。
        - 返回值 >= 0：
          * 若 p 在矩形外：返回 p 到矩形边界最近点的距离
          * 若 p 在矩形内：返回 0（因为点已经进入矩形内部，离边界的“外部距离”视为 0）
        """
        # 1) 把点从世界坐标变到矩形自身坐标系（相当于把矩形“转正”成轴对齐）
        dx, dy = float(p[0] - c[0]), float(p[1] - c[1])
        cy, sy = math.cos(yaw), math.sin(yaw)
        # world -> local: R^T (p - c)
        lx =  cy * dx + sy * dy
        ly = -sy * dx + cy * dy

        # 2) 在矩形局部坐标下，计算点到轴对齐矩形(AABB)的外部距离
        #    ax/ay 表示点在 x/y 方向上超出矩形边界多少（<=0 表示在内部投影范围内）
        ax = abs(lx) - half_l
        ay = abs(ly) - half_w

        # 只关心“超出”的部分（在内部则当作 0）
        ox = max(ax, 0.0)
        oy = max(ay, 0.0)

        # 点到矩形外部最近点的欧式距离
        return float(math.hypot(ox, oy))

    def _min_surface_clearance_to_obstacles(self, rid: int) -> float:
        """
        计算机器人 rid 到“静态障碍物（circle/rectangle）”的最小剩余距离（单位：米）。

        返回值含义（非常直观）：
        - 返回的是“机器人外壳”到“障碍物外边界”之间还剩多少米
        - 0 表示：已经碰到/重叠（相交）
        - >0 表示：还没碰到，数值越大离得越远

        重要：
        - 只遍历 self.obstacle_list（不会把其他机器人算进去）
        - 机器人在这里按圆形处理，半径为 self.robot_radius
        """
        p = self.robot_list[rid].state[:2].ravel().astype(np.float32)  # 机器人中心
        r_robot = float(self.robot_radius)                             # 机器人“外壳半径”

        best = float("inf")

        for obs in self.obstacle_list:
            # 兼容：shape 可能是 dict 或对象
            shape = getattr(obs, "shape", None)
            if shape is None:
                continue

            if isinstance(shape, dict):
                name   = shape.get("name", "")
                radius = shape.get("radius", None)
                length = shape.get("length", None)
                width  = shape.get("width", None)
            else:
                name   = getattr(shape, "name", "")
                radius = getattr(shape, "radius", None)
                length = getattr(shape, "length", None)
                width  = getattr(shape, "width", None)

            st = getattr(obs, "state", None)
            if st is None:
                continue
            cx, cy = float(st[0]), float(st[1])
            yaw = float(st[2]) if len(st) >= 3 else 0.0
            c = np.array([cx, cy], dtype=np.float32)

            # ---------------- 圆形障碍 ----------------
            if name == "circle":
                if radius is None:
                    continue
                # 中心距
                d_center = float(np.linalg.norm(p - c))
                # 外壳到外壳剩余距离 = 中心距 - (障碍半径 + 机器人半径)
                d = d_center - (float(radius) + r_robot)
                # 若发生重叠/相交，按 0（表示已经碰到）
                d = max(0.0, d)

            # ---------------- 矩形障碍（可带朝向 yaw） ----------------
            elif name == "rectangle":
                if length is None or width is None:
                    continue
                half_l = 0.5 * float(length)
                half_w = 0.5 * float(width)

                # 先算“机器人中心点”到矩形边界的最短距离
                d_point_to_rect = self._dist_point_to_obb_boundary(p, c, half_l, half_w, yaw)

                # 再把机器人半径扣掉 -> 得到“机器人外壳”到“矩形边界”还剩多少
                d = d_point_to_rect - r_robot
                d = max(0.0, d)

            else:
                continue

            if d < best:
                best = d

        # 没有障碍物时，返回一个较大的值（用于避免 inf 传播）
        if best == float("inf"):
            best = float(self.r_max)

        return float(best)

    def _min_surface_clearance_to_walls(self, rid: int) -> float:
        """
        计算机器人 rid 到“环境边界墙”的最小剩余距离（单位：米）。

        解释：
        - 先算机器人中心到四面墙的距离：左/右/下/上
        - 取最小的那一个（离最近墙有多远）
        - 再减机器人半径 -> 得到机器人外壳离墙还剩多少
        - <=0 视为 0（已经贴墙/越界）
        """
        p = self.robot_list[rid].state[:2].ravel().astype(np.float32)
        r_robot = float(self.robot_radius)

        xmin, xmax = float(self.x_range[0]), float(self.x_range[1])
        ymin, ymax = float(self.y_range[0]), float(self.y_range[1])
        x, y = float(p[0]), float(p[1])

        # 中心到四面墙的距离
        d_left   = x - xmin
        d_right  = xmax - x
        d_bottom = y - ymin
        d_top    = ymax - y

        d_center_to_wall = min(d_left, d_right, d_bottom, d_top)

        # 外壳到墙的剩余距离
        d = d_center_to_wall - r_robot
        return float(max(0.0, d))

    def _min_surface_clearance_static(self, rid: int):
        """
        综合输出（静态障碍 + 边界墙）：

        返回：
        - min_clear   : 机器人外壳到“最近的静态障碍或墙”还剩多少米
        - clear_obs   : 机器人外壳到“最近静态障碍”还剩多少米
        - clear_wall  : 机器人外壳到“最近边界墙”还剩多少米

        用法：
        - reward 里通常用 min_clear 做“贴障碍/贴墙风险”
        - 如果你想区分是贴墙还是贴障碍，可看 clear_obs / clear_wall
        """
        clear_obs  = self._min_surface_clearance_to_obstacles(rid)
        clear_wall = self._min_surface_clearance_to_walls(rid)
        return min(clear_obs, clear_wall), clear_obs, clear_wall

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
    
    def _compute_travel_time_and_prior_dir(self, pos: np.ndarray, goal: np.ndarray):
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
            return d, prior_dir

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
        gn = torch.norm(grad_start, dim=-1, keepdim=True)

        # fallback：梯度太小就用“指向 goal”的方向
        if float(gn.item()) < 1e-8:
            diff = (goal_norm - pos_norm).astype(np.float32)
            n = float(np.linalg.norm(diff))
            prior = (diff / (n + 1e-8)).astype(np.float32)
        else:
            prior_t = grad_start / (gn + 1e-8)  # [1,2]
            prior = prior_t.detach().cpu().numpy().astype(np.float32).reshape(-1)
        
        return float(TT.item()), prior
    
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
        cfg = self.reward_cfg

        # 三个半径门槛
        d_danger = 2.0 * r + cfg.peer_danger_offset
        d_safe   = d_danger + cfg.peer_safe_margin
        d_sense  = d_safe   + cfg.peer_sense_margin

        # 三段对应的大致势场强度
        k_far  = cfg.peer_k_far
        k_mid  = cfg.peer_k_mid
        k_max  = cfg.peer_k_max

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
        cfg = self.reward_cfg

        d_danger = 2.0 * r + cfg.peer_danger_offset
        d_safe   = d_danger + cfg.peer_safe_margin

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
        beta = cfg.peer_potential_beta
        m = float(arr.max())
        Phi = m + (1.0 / beta) * float(np.log(np.mean(np.exp(beta * (arr - m)))))

        return float(Phi), min_dist, in_peer_danger

    def _calculate_reward(
        self,
        id: int,
        action_raw,
        prior_dir_prev: np.ndarray,
        exec_dir: np.ndarray,
        exec_speed: float,
        is_collision: bool,
        is_arrive: bool,
        is_truncated: bool,
    ):
        robot = self.robot_list[id]
        cfg = self.reward_cfg

        # ---------------- 权重（核心：让 shaping 变成正的） ----------------
        w_T_progress   = cfg.w_T_progress
        w_step         = cfg.w_step

        w_peer_delta   = cfg.w_peer_delta
        w_peer_static  = cfg.w_peer_static

        w_obst_delta   = cfg.w_obst_delta
        w_obst_static  = cfg.w_obst_static

        w_speed_safe   = cfg.w_speed_safe
        w_speed_danger = cfg.w_speed_danger

        w_prior_align  = cfg.w_prior_align

        bonus_arrive           = cfg.bonus_arrive
        pen_timeout            = cfg.pen_timeout
        pen_collision_robot    = cfg.pen_collision_robot
        pen_collision_obstacle = cfg.pen_collision_obstacle

        r = float(self.robot_radius)

        # ---------------- 当前几何量 ----------------
        robot_pos  = robot.state[:2].ravel()
        robot_goal = robot.goal[:2].ravel()

        speed   = float(exec_speed)
        cur_dir = exec_dir.astype(np.float32).reshape(-1)

        # ===================== 1) NTFields 时间进度 + 当前先验方向 =====================
        T_curr, prior_dir_curr = self._compute_travel_time_and_prior_dir(robot_pos, robot_goal)
        T_prev = float(self.robot_last_T[id].item())
        delta_T = float(np.clip(T_prev - T_curr, -cfg.delta_T_clip, cfg.delta_T_clip))

        # ===================== 2) 多机器人势场 Φ 与 ΔΦ =====================
        Phi_curr, min_peer_dist, in_peer_danger = self._compute_peer_potential(id)
        Phi_prev = float(self.robot_last_peer_phi[id].item())
        delta_Phi = float(Phi_curr - Phi_prev)

        # ===================== 3) 障碍物/墙清距（不用 lidar） =====================
        min_clear, clear_obs, clear_wall = self._min_surface_clearance_static(id)

        clear_danger = cfg.clear_danger
        clear_safe   = cfg.clear_safe

        if min_clear <= clear_danger:
            obst_risk = 1.0
        elif min_clear >= clear_safe:
            obst_risk = 0.0
        else:
            obst_risk = float((clear_safe - min_clear) / max(clear_safe - clear_danger, 1e-6))
            obst_risk = float(np.clip(obst_risk, 0.0, 1.0))

        in_obst_danger = (obst_risk > 1e-6)

        loss_clear = float(obst_risk)
        loss_prev  = float(self.robot_last_obst_loss[id].item())
        delta_loss = float(loss_clear - loss_prev)
        self.robot_last_obst_loss[id] = float(loss_clear)

        # ===================== 4) 风险 gate =====================
        d_danger = 2.0 * r + cfg.peer_danger_offset
        d_safe   = d_danger + cfg.peer_safe_margin

        if min_peer_dist <= d_danger:
            peer_risk = 1.0
        elif min_peer_dist >= d_safe:
            peer_risk = 0.0
        else:
            peer_risk = float((d_safe - min_peer_dist) / max(d_safe - d_danger, 1e-6))
            peer_risk = float(np.clip(peer_risk, 0.0, 1.0))

        risk = max(peer_risk, obst_risk)
        # 关键：不要让 progress_gate 直接变 0（避免“原地等死”）
        progress_gate = float(np.clip(1.0 - cfg.progress_gate_risk_coeff * risk, cfg.progress_gate_min, 1.0))
        danger_gate   = float(risk)

        # ===================== 5) reward 组合 =====================
        reward = 0.0

        # (a) 时间进度
        r_T_progress = w_T_progress * progress_gate * delta_T
        reward += r_T_progress

        # (b) 每步时间成本
        r_step = -w_step
        reward += r_step

        # (c) 多机器人势场：ΔΦ + 静态 Φ（只在避让区）
        w_peer_away = cfg.w_peer_away_factor * w_peer_delta  # ★ 增强“远离别人”的收益，避免停滞
        r_peer_delta  = -w_peer_delta * max(delta_Phi, 0.0) + w_peer_away * max(-delta_Phi, 0.0)
        r_peer_static = -w_peer_static * Phi_curr if in_peer_danger else 0.0
        reward += r_peer_delta + r_peer_static

        # (d) 障碍物清距：Δloss + 静态 loss（只在有风险时）
        if in_obst_danger:
            w_obst_away = cfg.w_obst_away_factor * w_obst_delta
            r_obst_delta  = -w_obst_delta * max(delta_loss, 0.0) + w_obst_away * max(-delta_loss, 0.0)
            r_obst_static = -w_obst_static * loss_clear
        else:
            r_obst_delta  = 0.0
            r_obst_static = 0.0
        reward += r_obst_delta + r_obst_static

        # (e) 速度项：安全区鼓励快，危险区惩罚快
        v_max = max(float(self.v_max), 1e-6)
        speed_norm = float(np.clip(speed / v_max, 0.0, 1.0))
        r_speed = (w_speed_safe * (1.0 - danger_gate) * speed_norm) - (w_speed_danger * danger_gate * (speed_norm ** 2))
        reward += r_speed

        # (f) 先验方向对齐（安全区更强）
        r_prior_align = 0.0
        align_gate = progress_gate * (speed_norm ** cfg.align_speed_power)
        pn = float(np.linalg.norm(prior_dir_prev))
        if pn > 1e-6 and speed > 1e-4:
            prior_unit = (prior_dir_prev / pn).astype(np.float32).reshape(-1)
            cos_theta = float(np.dot(cur_dir, prior_unit))
            cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
            r_prior_align = w_prior_align * cos_theta
            reward += r_prior_align * align_gate

        # (g) 终止项
        r_arrive  = bonus_arrive if is_arrive else 0.0
        r_timeout = -pen_timeout if is_truncated else 0.0

        if is_collision:
            if min_peer_dist < 2.0 * self.robot_radius + cfg.collision_peer_margin:
                r_coll_robot = -pen_collision_robot
                r_coll_obst  = 0.0
            else:
                r_coll_robot = 0.0
                r_coll_obst  = -pen_collision_obstacle
        else:
            r_coll_robot = 0.0
            r_coll_obst  = 0.0

        reward += r_arrive + r_timeout + r_coll_robot + r_coll_obst

        # ===================== 6) 更新缓存 =====================
        self.robot_last_T[id]         = float(T_curr)
        self.robot_last_peer_phi[id]  = float(Phi_curr)
        self.robot_last_prior_dir[id] = prior_dir_curr.astype(np.float32)

        info = {
            "r_total":         float(reward),
            "r_T_progress":    float(r_T_progress),
            "r_step":          float(r_step),
            "r_peer_delta":    float(r_peer_delta),
            "r_peer_static":   float(r_peer_static),
            "r_obst_delta":    float(r_obst_delta),
            "r_obst_static":   float(r_obst_static),
            "r_speed":         float(r_speed),
            "r_prior_align_raw": float(r_prior_align),
            "r_prior_align":   float(r_prior_align * align_gate),
            "r_arrive":        float(r_arrive),
            "r_timeout":       float(r_timeout),
            "r_coll_robot":    float(r_coll_robot),
            "r_coll_obst":     float(r_coll_obst),

            "T_curr":          float(T_curr),
            "delta_T":         float(delta_T),
            "Phi_curr":        float(Phi_curr),
            "delta_Phi":       float(delta_Phi),
            "min_peer_dist":   float(min_peer_dist),

            "min_clear":       float(min_clear),
            "clear_obs":       float(clear_obs),
            "clear_wall":      float(clear_wall),
            "peer_risk":       float(peer_risk),
            "obst_risk":       float(obst_risk),

            "speed":           float(speed),
            "speed_norm":      float(speed_norm),
            "in_peer_danger":  bool(in_peer_danger),
            "in_obst_danger":  bool(in_obst_danger),
        }

        return float(reward), info

    def reset(self, robot_states=None, robot_goals=None):
        if robot_states is not None:
            assert len(robot_states) == self.num_robots
        if robot_goals is not None:
            assert len(robot_goals) == self.num_robots
        
        self.start_goal_cache = []
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
        self.robot_last_obst_loss = torch.zeros(self.num_robots, dtype=torch.float32)

        positions = []
        goals = []
        for robot in self.robot_list:
            positions.append(robot.state[:2].ravel())
            goals.append(robot.goal[:2].ravel())

        positions = np.asarray(positions, dtype=np.float32)
        goals = np.asarray(goals, dtype=np.float32)

        # 初始化每个机器人的初始 goal 距离
        # 初始化Time Field时间缓存 T_i(0) 和势场 Φ_i(0)
        for i in range(self.num_robots):
            dis = np.linalg.norm(goals[i] - positions[i])
            self.robot_last_goal_dist[i] = float(dis)

            min_clear, _, _ = self._min_surface_clearance_static(i)
            self.robot_last_obst_loss[i] = min_clear

            T0, prior0 = self._compute_travel_time_and_prior_dir(positions[i], goals[i])
            self.robot_last_T[i] = float(T0)
            self.robot_last_prior_dir[i] = prior0

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
        robot_goal=None,
        dis_tol=1.5,
        max_attempts=200
    ):
        def is_valid_pos(pos):
            for sg in self.start_goal_cache:
                if np.linalg.norm(pos - sg) < dis_tol:
                    return False
            return True
        
        if robot_state is not None:
            robot.set_state(np.array(robot_state), init=True)
        else:
            for _ in range(max_attempts):
                free_pos = robot.get_random_free_position(
                    self.env.objects,
                    free = True,
                    filter_start_goal = True,
                    range_limits=[
                        [self.x_range[0] + 1, self.y_range[0] + 1, 0],
                        [self.x_range[1] - 1, self.y_range[1] - 1, 0],
                    ],
                )
                if is_valid_pos(free_pos):
                    self.start_goal_cache.append(free_pos)
                    break
            robot.set_state(free_pos, init=True)
            
        if robot_goal is not None:
            robot.set_goal(np.array(robot_goal), init=True)
        else:
            for _ in range(max_attempts):
                free_pos = robot.get_random_free_position(
                    self.env.objects,
                    free = True,
                    filter_start_goal = True,
                    range_limits=[
                        [self.x_range[0] + 1, self.y_range[0] + 1, 0],
                        [self.x_range[1] - 1, self.y_range[1] - 1, 0],
                    ],
                )
                if is_valid_pos(free_pos):
                    self.start_goal_cache.append(free_pos)
                    break
            robot.set_goal(free_pos, init=True)
        # 重置碰撞等信息
        robot.reset()