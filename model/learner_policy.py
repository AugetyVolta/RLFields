"""Partial GPU-tower synchronization for the RLlib Policy/ModelV2 stack.

The SGD body is adapted from Ray 2.51.1 (Apache-2.0):
https://github.com/ray-project/ray/blob/ray-2.51.1/rllib/policy/torch_policy_v2.py
PPO loss and optimizer order are inherited unchanged.
"""
import torch
from ray.rllib.algorithms import registry as policy_registry
from ray.rllib.algorithms.ppo import ppo_torch_policy
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy.torch_policy import _directStepOptimizerSingleton
from ray.rllib.utils.metrics import NUM_GRAD_UPDATES_LIFETIME, DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY


def restore_optimizer_step_devices(policy):
    # Native Adam keeps its scalar step on CPU unless capturable/fused is enabled.
    # Ray's generic restore moves it to CUDA along with the moment tensors.
    for optimizer in policy._optimizers:
        if not isinstance(optimizer, torch.optim.Adam):
            continue
        for group in optimizer.param_groups:
            if group.get("capturable") or group.get("fused"):
                continue
            for parameter in group["params"]:
                state = optimizer.state.get(parameter, {})
                if isinstance(state.get("step"), torch.Tensor):
                    state["step"] = state["step"].cpu()


class RLFieldsPPOTorchPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        self.sync_trainable_only = True
        self._frozen_towers_ready = False
        self.sync_stats = {"full_sync_bytes": 0, "step_sync_bytes": 0}
        super().__init__(observation_space, action_space, config)

    def set_weights(self, weights):
        super().set_weights(weights)
        # A restored planner needs one complete copy before partial synchronization.
        self._frozen_towers_ready = False

    def set_state(self, state):
        super().set_state(state)
        restore_optimizer_step_devices(self)
        # Ray saves this counter but its legacy Policy.set_state does not restore it.
        if "num_grad_updates" in state:
            self.num_grad_updates = int(state["num_grad_updates"])
        self._frozen_towers_ready = False

    def _sync_towers(self, *, force_full=False):
        if len(self.devices) <= 1:
            self._frozen_towers_ready = True
            return
        assert self.model_gpu_towers[0] is self.model
        state = self.model.state_dict()
        full = force_full or not self.sync_trainable_only or not self._frozen_towers_ready
        if full:
            sync_state = state
            frozen = set()
        else:
            frozen = {name for name, parameter in self.model.named_parameters()
                      if not parameter.requires_grad}
            # All buffers remain in the synchronization, including running statistics.
            sync_state = {name: value for name, value in state.items() if name not in frozen}
        for tower in self.model_gpu_towers[1:]:
            incompatible = tower.load_state_dict(sync_state, strict=full)
            if not full and (set(incompatible.missing_keys) != frozen or incompatible.unexpected_keys):
                raise RuntimeError("Partial tower synchronization omitted non-frozen state")
        key = "full_sync_bytes" if full else "step_sync_bytes"
        self.sync_stats[key] += sum(t.numel() * t.element_size() for t in sync_state.values()) * (len(self.devices) - 1)
        self._frozen_towers_ready = True

    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0):
        if len(self.devices) <= 1:
            return super().learn_on_loaded_batch(offset, buffer_index)
        if not self._loaded_batches[buffer_index]:
            raise ValueError(
                "Must call Policy.load_batch_into_buffer() before "
                "Policy.learn_on_loaded_batch()!"
            )

        # Get the correct slice of the already loaded batch to use,
        # based on offset and batch size.
        device_batch_size = self.config.get("minibatch_size")
        if device_batch_size is None:
            device_batch_size = self.config.get(
                "sgd_minibatch_size",
                self.config["train_batch_size"],
            )
        device_batch_size //= len(self.devices)

        # Set Model to train mode.
        if self.model_gpu_towers:
            for t in self.model_gpu_towers:
                t.train()

        self._sync_towers()

        if device_batch_size >= sum(len(s) for s in self._loaded_batches[buffer_index]):
            device_batches = self._loaded_batches[buffer_index]
        else:
            device_batches = [
                b[offset : offset + device_batch_size]
                for b in self._loaded_batches[buffer_index]
            ]

        # Callback handling.
        batch_fetches = {}
        for i, batch in enumerate(device_batches):
            custom_metrics = {}
            self.callbacks.on_learn_on_batch(
                policy=self, train_batch=batch, result=custom_metrics
            )
            batch_fetches[f"tower_{i}"] = {"custom_metrics": custom_metrics}

        # Do the (maybe parallelized) gradient calculation step.
        tower_outputs = self._multi_gpu_parallel_grad_calc(device_batches)

        # Mean-reduce gradients over GPU-towers (do this on CPU: self.device).
        all_grads = []
        for i in range(len(tower_outputs[0][0])):
            if tower_outputs[0][0][i] is not None:
                all_grads.append(
                    torch.mean(
                        torch.stack([t[0][i].to(self.device) for t in tower_outputs]),
                        dim=0,
                    )
                )
            else:
                all_grads.append(None)
        # Set main model's grads to mean-reduced values.
        for i, p in enumerate(self.model.parameters()):
            p.grad = all_grads[i]

        self.apply_gradients(_directStepOptimizerSingleton)

        self.num_grad_updates += 1

        for i, (model, batch) in enumerate(zip(self.model_gpu_towers, device_batches)):
            batch_fetches[f"tower_{i}"].update(
                {
                    LEARNER_STATS_KEY: self.stats_fn(batch),
                    "model": model.metrics(),
                    NUM_GRAD_UPDATES_LIFETIME: self.num_grad_updates,
                    # -1, b/c we have to measure this diff before we do the update
                    # above.
                    DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY: (
                        self.num_grad_updates - 1 - (batch.num_grad_updates or 0)
                    ),
                }
            )
        batch_fetches.update(self.extra_compute_grad_fetches())

        return batch_fetches


# The legacy policy registry resolves names inside ray.rllib.algorithms.
# Expose our additional class name while retaining the native PPO policy.
ppo_torch_policy.RLFieldsPPOTorchPolicy = RLFieldsPPOTorchPolicy
policy_registry.POLICIES["RLFieldsPPOTorchPolicy"] = "ppo.ppo_torch_policy"
