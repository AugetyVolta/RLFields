# Third-party notices

The MIT license in the repository root applies to original PGRL-Nav source code.
The following third-party components retain their upstream licenses.

## PC-Planner

- Upstream: https://github.com/fishcrap/pc-planner
- License: MIT; [full license and original copyright notice](licenses/PC-Planner-MIT.txt).
- Copyright (c) 2024 xvjie shen.

PGRL-Nav uses PC-Planner as its neural global-planning prior. The upstream
license notice is retained here alongside the paper citation in the README.

## Ray / RLlib

- Upstream version: Ray 2.51.1.
- Source: https://github.com/ray-project/ray/blob/ray-2.51.1/rllib/policy/torch_policy_v2.py
- License: [Apache License 2.0](licenses/Apache-2.0.txt).
- Affected file: `model/learner_policy.py`.

The SGD implementation in `model/learner_policy.py` is adapted from RLlib's
`TorchPolicyV2`. PGRL-Nav modifies GPU-tower synchronization to reuse frozen
planner parameters while retaining the upstream PPO loss and optimizer order.
The Ray-derived portions remain subject to Apache License 2.0; the root MIT
license does not replace those terms.

Other installed dependencies retain their own licenses, distributed with their
respective packages.
