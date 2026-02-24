import math
import numpy as np

from .marl_sim_env import MARL_SIM
from .esdf_2d_safe import ESDF_2d_Safe


class MARL_SIM_ActionDirect_PriorAlign_NoPriorAlignReward(MARL_SIM):
    """
    no-prior-align 观测基线上的 direct-action + no-prior-align-reward 消融：
    - 动作语义：direct (alpha, theta_global)
    - 观测语义：不做 prior 旋转（SDF/neighbor 维持原始坐标）
    - 奖励：仅关闭 w_prior_align，其他保持与主奖励一致
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 兼容空点云
        self.esdf = ESDF_2d_Safe(self.esdf_cfg)

    def step(self, action_dict):
        applied_actions = []
        action_ids = []
        raw_actions = {}
        exec_dirs = {}
        exec_speeds = {}

        # 1) direct action: (alpha, theta_global) -> (dx, dy)
        for robot_id, action in action_dict.items():
            rid = int(robot_id.split("_")[-1])
            action_ids.append(rid)

            alpha_raw = float(action[0])
            theta_raw = float(action[1])
            raw_actions[rid] = np.array(action, dtype=np.float32)

            alpha01 = (alpha_raw + 1.0) / 2.0
            alpha01 = float(np.clip(alpha01, 0.0, 1.0))

            theta = float(theta_raw) * math.pi
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

        # 2) reward + obs（obs 使用 MARL_SIM 原始版本，不做 prior 旋转）
        for rid in action_ids:
            robot_id = f"robot_{rid}"

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

    def _calculate_reward(
        self,
        id: int,
        action_raw,
        prior_dir_prev,
        exec_dir,
        exec_speed: float,
        is_collision: bool,
        is_arrive: bool,
        is_truncated: bool,
    ):
        cfg = self.reward_cfg
        old = float(cfg.w_prior_align)
        cfg.w_prior_align = 0.0
        try:
            return MARL_SIM._calculate_reward(
                self,
                id=id,
                action_raw=action_raw,
                prior_dir_prev=prior_dir_prev,
                exec_dir=exec_dir,
                exec_speed=exec_speed,
                is_collision=is_collision,
                is_arrive=is_arrive,
                is_truncated=is_truncated,
            )
        finally:
            cfg.w_prior_align = old
