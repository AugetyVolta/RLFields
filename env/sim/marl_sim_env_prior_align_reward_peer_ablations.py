import numpy as np

from .marl_sim_env_prior_align import MARL_SIM_PriorAlign


class MARL_SIM_PriorAlign_NoPeerPotentialReward(MARL_SIM_PriorAlign):
    """
    Reward ablation on prior-align baseline:
    disable peer potential terms only (r_peer_delta, r_peer_static).
    """

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
        old_w_peer_delta = float(cfg.w_peer_delta)
        old_w_peer_static = float(cfg.w_peer_static)
        cfg.w_peer_delta = 0.0
        cfg.w_peer_static = 0.0
        try:
            return MARL_SIM_PriorAlign._calculate_reward(
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
            cfg.w_peer_delta = old_w_peer_delta
            cfg.w_peer_static = old_w_peer_static


class MARL_SIM_PriorAlign_PeerNearestReward(MARL_SIM_PriorAlign):
    """
    Reward ablation on prior-align baseline:
    replace smooth-max peer aggregation with nearest-neighbor potential.
    """

    def _compute_peer_potential(self, id: int):
        robot_pos = self.robot_list[id].state[:2].ravel()
        r = float(self.robot_radius)
        cfg = self.reward_cfg

        d_danger = 2.0 * r + cfg.peer_danger_offset
        d_safe = d_danger + cfg.peer_safe_margin

        min_dist = float("inf")
        for j, other in enumerate(self.robot_list):
            if j == id:
                continue
            other_pos = other.state[:2].ravel()
            d = float(np.linalg.norm(other_pos - robot_pos))
            if d < min_dist:
                min_dist = d

        if min_dist == float("inf"):
            return 0.0, min_dist, False

        in_peer_danger = bool(min_dist < d_safe)
        phi = self._single_neighbor_potential(min_dist)
        return float(phi), float(min_dist), in_peer_danger
