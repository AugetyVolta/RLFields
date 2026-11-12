"""Copy a project checkpoint into the current module/config naming scheme."""
import argparse
import copy
from dataclasses import fields
import json
from pathlib import Path
import pickle
import shlex
import shutil
import sys

import cloudpickle
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.params import SceneConfig, RewardConfig
from env.multi_robot_env import MultiRobotEnv
from env.sim.marl_sim_env import MARL_SIM
from model.models import RLFields
from model.policy_value_network import RLFieldsActorModel
import yaml


class CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Recover project class families into their current canonical modules.
        families = {
            'env.multi_robot_env': MultiRobotEnv,
            'env.sim.marl_sim_env': MARL_SIM,
            'model.models': RLFields,
            'model.policy_value_network': RLFieldsActorModel,
        }
        for base, target in families.items():
            if (module == base or module.startswith(base + '_')) and name.startswith(target.__name__):
                return target
        return super().find_class(module, name)


def normalize(value, renames):
    if isinstance(value, AlgorithmConfig):
        value = value.to_dict()
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key = renames.get(key, key) if isinstance(key, str) else key
            if key in result:
                raise ValueError(f'Duplicate key after renaming: {key}')
            result[key] = normalize(item, renames)
        if isinstance(result.get('model'), dict) and result['model'].get('custom_model'):
            result['model']['custom_model'] = 'rlfields_actor'
        return result
    if isinstance(value, list):
        return [normalize(item, renames) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize(item, renames) for item in value)
    return value


def bind_config(value, env_config, network_config):
    if isinstance(value, dict):
        for item in value.values():
            bind_config(item, env_config, network_config)
        if 'env_config' in value and isinstance(value.get('model'), dict):
            value['env'] = MultiRobotEnv
            value['env_config'] = copy.deepcopy(env_config)
            value['model']['custom_model'] = 'rlfields_actor'
            value['model']['custom_model_config'] = copy.deepcopy(network_config)
    elif isinstance(value, (list, tuple)):
        for item in value:
            bind_config(item, env_config, network_config)


def find_asset(run_dir, archive_name, recorded_path):
    candidates = [run_dir / archive_name] if run_dir else []
    if recorded_path:
        path = Path(recorded_path).expanduser()
        candidates += [path, Path(__file__).resolve().parents[1] / path]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f'Missing original asset: {archive_name}')


def migrate_checkpoint(checkpoint, output_dir, renames=None):
    checkpoint = Path(checkpoint).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    renames = {} if renames is None else renames
    if not checkpoint.is_dir():
        raise FileNotFoundError(f'Checkpoint directory not found: {checkpoint}')
    if output_dir == checkpoint or checkpoint in output_dir.parents:
        raise ValueError('Output must be outside the source checkpoint directory.')
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f'Output directory must be absent or empty: {output_dir}')
    states = {}
    for path in checkpoint.rglob('*.pkl'):
        with path.open('rb') as f:
            states[path.relative_to(checkpoint)] = normalize(CheckpointUnpickler(f).load(), renames)
    if Path('algorithm_state.pkl') not in states or Path('policies/shared_policy/policy_state.pkl') not in states:
        raise ValueError('Expected an Algorithm checkpoint containing shared_policy state.')
    algorithm = states[Path('algorithm_state.pkl')]
    policy = states[Path('policies/shared_policy/policy_state.pkl')]
    if 'fused_ln.weight' not in policy['weights']:
        raise ValueError('Checkpoint architecture is incompatible with the current policy model.')
    iteration = int(algorithm['training_iteration'])
    if iteration < 1:
        raise ValueError('Checkpoint must contain at least one completed training iteration.')
    config = policy['policy_spec']['config']
    scene = config.get('env_config', {}).get('scene_config', {})
    run_dir = None
    for parent in checkpoint.parents:
        snapshot = parent / 'scene_config.yaml'
        if not snapshot.is_file():
            snapshot = parent / 'scene_config_resume.yaml'
        if snapshot.is_file():
            run_dir = parent
            if not scene:
                with snapshot.open(encoding='utf-8') as f:
                    scene = normalize(yaml.safe_load(f) or {}, renames)
            break
    if not scene:
        raise ValueError('Original scene configuration is required to preserve training settings.')
    unknown = set(scene.get('reward_params', {})) - {item.name for item in fields(RewardConfig)}
    if unknown:
        raise ValueError(f'Unrecognized reward keys {sorted(unknown)}; specify --rename-key OLD=NEW explicitly.')
    env_source = find_asset(run_dir, 'env.yaml', config.get('env_config', {}).get('env_yaml_path')
                            or scene.get('experiment_params', {}).get('env_yaml_path'))
    planner_source = find_asset(run_dir, 'ntfields.pt', config['model'].get('custom_model_config', {}).get('nt_pretrained_path')
                                or scene.get('network_params', {}).get('nt_pretrained_path'))
    cfg = SceneConfig.from_dict(scene)
    for item in fields(cfg.algorithm_params):
        key = 'lambda' if item.name == 'lambda_' else item.name
        if key in config:
            setattr(cfg.algorithm_params, item.name, config[key])
    for key, value in config['model'].get('custom_model_config', {}).items():
        if hasattr(cfg.network_params, key):
            setattr(cfg.network_params, key, value)
    env_config = config.get('env_config', {})
    cfg.robot_params.number_robots = env_config.get('num_robots', cfg.robot_params.number_robots)
    cfg.network_params.num_use_neighbor_obs = env_config.get('num_use_neighbor_obs', cfg.network_params.num_use_neighbor_obs)
    cfg.esdf_params.map_size = env_config.get('H', cfg.esdf_params.map_size)
    cfg.esdf_params.resolution = (cfg.esdf_params.x_max - cfg.esdf_params.x_min) / cfg.esdf_params.map_size
    cfg.experiment_params.exp_dir = str(output_dir)
    cfg.experiment_params.env_yaml_path = str(env_source)
    cfg.network_params.nt_pretrained_path = str(planner_source)
    cfg._environment_info.setdefault('name', Path(scene.get('experiment_params', {}).get('env_yaml_path') or env_source).stem)
    cfg._environment_info.update(migration_source=str(checkpoint), source_yaml=str(env_source), source_planner=str(planner_source))
    # No weights, optimizer values, timesteps or random states are changed.
    cfg.get_simulation_config()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / 'ckpts' / f'iter_{iteration:06d}'
    shutil.copytree(checkpoint, target)
    shutil.copy2(env_source, output_dir / 'env.yaml')
    shutil.copy2(planner_source, output_dir / 'ntfields.pt')
    (output_dir / 'tensorboard').mkdir()
    cfg.experiment_params.env_yaml_path = str(output_dir / 'env.yaml')
    cfg.network_params.nt_pretrained_path = str(output_dir / 'ntfields.pt')
    cfg.dump(str(output_dir / 'scene_config.yaml'))
    for value in states.values():
        bind_config(value, cfg.get_env_config(), cfg.get_network_config())
    for relative, value in states.items():
        with (target / relative).open('wb') as f:
            cloudpickle.dump(value, f)
    # Infer feature switches from the saved tensor structure, then archive the resolved config.
    cfg = SceneConfig.from_checkpoint(str(target))
    cfg.experiment_params.exp_dir = str(output_dir)
    cfg.dump(str(output_dir / 'scene_config.yaml'))
    for value in states.values():
        bind_config(value, cfg.get_env_config(), cfg.get_network_config())
    for relative, value in states.items():
        with (target / relative).open('wb') as f:
            cloudpickle.dump(value, f)
    for metadata in target.rglob('rllib_checkpoint.json'):
        data = json.loads(metadata.read_text())
        if data.get('state_file'):
            data['state_file'] = Path(data['state_file']).name
        metadata.write_text(json.dumps(data, indent=2))
    return target, iteration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Original RLlib checkpoint directory')
    parser.add_argument('--output-dir', required=True, help='New experiment directory; original files stay unchanged')
    parser.add_argument('--rename-key', action='append', default=[], metavar='OLD=NEW', help='Explicitly rename a saved configuration key')
    args = parser.parse_args()
    renames = {}
    for item in args.rename_key:
        old, separator, new = item.partition('=')
        if not separator or not old or not new:
            parser.error('--rename-key must be OLD=NEW')
        renames[old] = new
    target, iteration = migrate_checkpoint(args.checkpoint, args.output_dir, renames)
    print(f'Converted checkpoint: {target}')
    print(shlex.join(['python', 'train.py', '--resume', str(target.parent.parent), '--iter', str(iteration)]))


if __name__ == '__main__':
    main()
