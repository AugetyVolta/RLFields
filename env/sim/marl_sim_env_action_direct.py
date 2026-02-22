import math
import numpy as np

from .marl_sim_env import MARL_SIM
from .esdf_2d_safe import ESDF_2d_Safe


class MARL_SIM_ActionDirect(MARL_SIM):
    """
    消融：policy 直接输出动作 (alpha, theta_global)，而不是“对 prior_dir 的调制”。

    - alpha_raw ∈ [-1,1] -> alpha01 ∈ [0,1] -> v = alpha01 * v_max
    - theta_raw ∈ [-1,1] -> theta ∈ [-pi, pi]
    - exec_dir = [cos(theta), sin(theta)]  (世界坐标系)

    额外：使用 ESDF_2d_Safe，避免点云为空时 np.min(empty) 报错。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 只替换 ESDF 生成器，不影响其他逻辑
        self.esdf = ESDF_2d_Safe(self.esdf_cfg)

    def step(self, action_dict):  # action_dict: dict(robot_id, [alpha, theta]), [-1, 1]
        # 这里沿用原 step 的结构，仅替换“动作->执行方向”的语义
        applied_actions = []
        action_ids = []
        raw_actions = {}   # 归一化动作 (alpha/theta)
        exec_dirs = {}     # 实际执行方向（单位向量）
        exec_speeds = {}   # 实际执行速度（m/s）

        # -------- 1) (alpha, theta_global) -> (dx, dy) 并执行 --------
        for robot_id, action in action_dict.items():
            rid = int(robot_id.split('_')[-1])  # 'robot_0' -> 0
            action_ids.append(rid)

            alpha_raw = float(action[0])
            theta_raw = float(action[1])
            raw_actions[rid] = np.array(action, dtype=np.float32)

            # alpha: [-1,1] -> [0,1]
            alpha01 = (alpha_raw + 1.0) / 2.0
            alpha01 = float(np.clip(alpha01, 0.0, 1.0))

            # theta: [-1,1] -> [-pi, pi]
            theta = float(theta_raw) * math.pi

            # 直接用世界系 heading
            exec_dir = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)

            v = alpha01 * float(self.v_max)
            dx = float(exec_dir[0]) * v * float(self.move_single_step)
            dy = float(exec_dir[1]) * v * float(self.move_single_step)

            applied_actions.append([dx, dy])
            exec_dirs[rid] = exec_dir
            exec_speeds[rid] = v

        self.env.step(action=applied_actions, action_id=action_ids)
        if self.render:
            self.env.render()
        self._env_step_id += 1

        obs = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}

        # -------- 2) reward + obs（完全复用父类逻辑）--------
        for rid in action_ids:
            robot_id = f'robot_{rid}'

            terminateds[robot_id], is_collision, is_arrive = self._is_terminated(rid)

            truncateds[robot_id] = False
            is_truncated = False
            if not self.robot_terminateds[rid]:
                self.robot_times[rid] += 1
                if self.robot_times[rid] >= self.single_max_timestep:
                    self.robot_truncateds[rid] = True
                    is_truncated = True
                    truncateds[robot_id] = self.robot_truncateds[rid].item()

            r, info = self._calculate_reward(
                id=rid,
                action_raw=raw_actions[rid],
                prior_dir_prev=self.robot_last_prior_dir[rid].copy(),
                exec_dir=exec_dirs[rid],
                exec_speed=exec_speeds[rid],
                is_collision=is_collision,
                is_arrive=is_arrive,
                is_truncated=is_truncated,
            )
            rewards[robot_id] = r
            infos[robot_id] = info
            infos[robot_id]["_env_step_id"] = self._env_step_id

            obs[robot_id] = self._prepare_obs(rid)

        robot_done = self.robot_terminateds | self.robot_truncateds
        done_all = bool(robot_done.all().item())
        all_naturally_terminated = bool(self.robot_terminateds.all().item())

        if done_all:
            if all_naturally_terminated:
                terminateds["__all__"] = True
                truncateds["__all__"] = False
            else:
                terminateds["__all__"] = False
                truncateds["__all__"] = True
        else:
            terminateds["__all__"] = False
            truncateds["__all__"] = False

        return obs, rewards, terminateds, truncateds, infos
