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

    # only alive robots have action
    def step(self, action_dict): # action_dict: dict of(robot_id, [d_x, d_y]), [-1, 1]
        applied_actions = []
        action_ids = []
        for robot_id, action in action_dict.items():
            id = int(robot_id.split('_')[-1]) # 'robot_0' -> 0
            action_ids.append(id)
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
            
            # 计算reward
            rewards[robot_id] = self._calculate_reward(
                id, is_collision, is_arrive, is_truncated
            )
            
            # # 对于终止或截断的机器人，重置计数器和标志
            # if is_collision or is_arrive or is_truncated:
            #     self.robot_times[id] = 0
            #     self.robot_terminateds[id] = False
            #     self.robot_truncateds[id] = False
            #     # 重置机器人位置和目标
            #     self.reset_robot(robot=self.robot_list[id])
            
            # 准备观测和info
            obs[robot_id] = self._prepare_obs(id)
            infos[robot_id] = {}
        
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
    
    def _calculate_reward(self, id: int, is_collision: bool, is_arrive: bool, is_truncated: bool):
        """
        稍微稠密一点的奖励设计：
        1) 距离进度 reward：朝目标前进给正奖励，远离扣分
        2) 每步小时间惩罚
        3) 碰撞大惩罚
        4) 到达大正奖励
        5) 超时额外惩罚
        """
        robot = self.robot_list[id]
        robot_pos = robot.state[:2].ravel()
        robot_goal = robot.goal[:2].ravel()
        dist = float(np.linalg.norm(robot_goal - robot_pos))   # 当前步距离（世界坐标）

        # 上一时刻的距离
        last_dist = float(self.robot_last_goal_dist[id].item())
        progress = last_dist - dist    # >0 表示靠近目标

        # 更新缓冲（放在一开始，以免后面早 return）
        self.robot_last_goal_dist[id] = dist

        reward = 0.0

        # 1) 距离进度项：系数可以调大/调小
        #    如果一步能靠近 0.05 米，系数设成 5.0，则该项 ~ +0.25
        reward += 5.0 * progress

        # 2) 每步时间惩罚（防止无限游荡）
        reward -= 0.01

        # 3) 碰撞惩罚
        if is_collision:
            reward -= 5.0

        # 4) 到达大正奖励
        if is_arrive:
            reward += 20.0

        # 5) 超时惩罚（被截断时）
        if is_truncated:
            reward -= 5.0

        return reward


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