import math
import numpy as np

from .marl_sim_env import MARL_SIM
from utils.utils import lidar_to_pointcloud


class MARL_SIM_PriorAlign(MARL_SIM):
    """
    Prior-aligned variant:
    - Lidar pointcloud rotated so prior_dir becomes +x axis.
    - Neighbor relative positions/goals rotated to the same prior-aligned frame.
    - self_obs absolute coords and prior_dir itself are kept unchanged.
    """

    @staticmethod
    def _rot_from_prior(prior_dir: np.ndarray) -> np.ndarray:
        p = np.asarray(prior_dir, dtype=np.float32).reshape(-1)
        if p.shape[0] < 2:
            return np.eye(2, dtype=np.float32)
        px, py = float(p[0]), float(p[1])
        n = math.hypot(px, py)
        if n < 1e-6:
            return np.eye(2, dtype=np.float32)
        c = px / n
        s = py / n
        # rotate world vectors into prior-aligned frame (prior -> +x)
        return np.array([[c, s], [-s, c]], dtype=np.float32)

    @staticmethod
    def _rot_vecs(vecs: np.ndarray, R: np.ndarray) -> np.ndarray:
        # vecs: (..., 2)
        return vecs @ R.T

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
            都是相对于当前机器人 (x,y) 的相对坐标（并旋转到 prior 对齐坐标系）
          - prior_dir 为 env 预先计算好的单位方向（与 policy 的 theta 旋转基准一致）
        """
        robot = self.robot_list[id]

        # ===== 1) self_obs: [x, y, goal_x, goal_y] (绝对坐标) =====
        robot_pos = robot.state[:self.self_obs_dim//2].ravel().astype(np.float32)  # (2,)
        robot_goal = robot.goal[:self.self_obs_dim//2].ravel().astype(np.float32) # (2,)
        robot_pos_norm  = (robot_pos  - self.world_center) / self.world_scale
        robot_goal_norm = (robot_goal - self.world_center) / self.world_scale
        self_obs = np.concatenate([robot_pos_norm, robot_goal_norm], axis=0)  # (self_obs_dim,)

        # ===== prior-aligned rotation =====
        prior_dir = self.robot_last_prior_dir[id].astype(np.float32)  # (2,)
        R = self._rot_from_prior(prior_dir)

        # ===== 2) neighbor_obs: 相对坐标 + 旋转到 prior 对齐坐标系 =====
        neighbor_feats = []
        for j, other in enumerate(self.robot_list):
            if j == id:
                continue
            other_pos = other.state[:2].ravel()
            other_goal = other.goal[:2].ravel()

            dist = np.linalg.norm(other_pos - robot_pos)
            rel_vec = other_pos - robot_pos
            if self.esdf_cfg.x_min <= rel_vec[0] <= self.esdf_cfg.x_max and \
                self.esdf_cfg.y_min <= rel_vec[1] <= self.esdf_cfg.y_max:
                rel_pos = other_pos - robot_pos
                rel_goal = other_goal - robot_pos

                # rotate to prior-aligned frame
                rel_pos = self._rot_vecs(rel_pos.reshape(1, 2), R)[0]
                rel_goal = self._rot_vecs(rel_goal.reshape(1, 2), R)[0]

                rel_pos_norm = rel_pos / self.world_scale
                rel_goal_norm = rel_goal / self.world_scale
                feat = np.concatenate([rel_pos_norm, rel_goal_norm], axis=0)
                neighbor_feats.append((dist, feat))

        neighbor_feats.sort(key=lambda x: x[0])

        K = self.num_use_neighbor_obs
        neighbor_obs = np.zeros((K, self.neighbor_obs_dim), dtype=np.float32)

        M = min(len(neighbor_feats), K)
        for k in range(M):
            neighbor_obs[k] = neighbor_feats[k][1].astype(np.float32)

        neighbor_valid_num = float(M)
        neighbor_obs_flat = neighbor_obs.reshape(-1)

        # ===== 3) ESDF/SDF 观测（点云旋转到 prior 对齐坐标系） =====
        pointcloud = lidar_to_pointcloud(robot.lidar.get_scan())
        if pointcloud.size > 0:
            pointcloud = self._rot_vecs(pointcloud, R)
        sdf_grid = self.esdf.generate_sdf_from_pointcloud(pointcloud)
        sdf_grid = np.clip(sdf_grid, 0.0, self.r_max) / self.r_max
        sdf_flat = sdf_grid.flatten()

        # ===== 4) prior action (direction) =====
        prior_dir_out = prior_dir.astype(np.float32)

        # ===== 5) 拼接总观测 =====
        obs = np.concatenate(
            [
                self_obs,
                neighbor_obs_flat,
                np.array([neighbor_valid_num], dtype=np.float32),
                sdf_flat,
                prior_dir_out,
            ],
            axis=0,
        )
        return obs.astype(np.float32)
