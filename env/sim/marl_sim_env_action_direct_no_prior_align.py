from .marl_sim_env import MARL_SIM
from .marl_sim_env_action_direct import MARL_SIM_ActionDirect


class MARL_SIM_ActionDirect_NoPriorAlign(MARL_SIM_ActionDirect):
    """Ablation-1/3 + reward ablation：动作 direct，并关闭 w_prior_align。"""

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
