from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import yaml

JAX_IMPORT_ERROR: Exception | None = None

try:
    import chex
    import distrax
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import struct
    from flax.linen.initializers import constant, orthogonal
    from flax.training import orbax_utils
    from flax.training.train_state import TrainState
    from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer

    from craftax.craftax_env import make_craftax_env_from_name
except Exception as exc:  # pragma: no cover - exercised only without optional deps
    JAX_IMPORT_ERROR = exc


TRACK_ID = "classic"
DEFAULT_CONFIG = "configs/classic_craftax_1m_random_init.yaml"


def now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must decode to an object: {config_path}")
    return payload


def ensure_dir(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def _import_status(module_name: str) -> dict[str, Any]:
    try:
        module = __import__(module_name)
    except Exception as exc:
        return {
            "module": module_name,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "module": module_name,
        "available": True,
        "version": getattr(module, "__version__", ""),
    }


def dependency_status() -> dict[str, Any]:
    required = [
        _import_status("jax"),
        _import_status("craftax"),
        _import_status("flax"),
        _import_status("optax"),
        _import_status("distrax"),
        _import_status("orbax"),
    ]
    optional = [_import_status("wandb")]
    return {
        "required": required,
        "optional": optional,
        "ready": all(item["available"] for item in required),
    }


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    task_cfg = config.get("task", {})
    env_cfg = config.get("environment", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    modal_cfg = config.get("modal", {})

    if str(task_cfg.get("track") or "").strip() != TRACK_ID:
        errors.append(f"task.track must be {TRACK_ID!r}")
    if str(env_cfg.get("package") or "").strip() != "craftax":
        errors.append("environment.package must be 'craftax'")
    if str(env_cfg.get("benchmark_variant") or "").strip().lower() != "1m":
        errors.append("environment.benchmark_variant must be '1m'")
    if not str(env_cfg.get("env_name") or "").strip():
        errors.append("environment.env_name must be set")
    if str(training_cfg.get("initialization") or "").strip().lower() != "random":
        errors.append("training.initialization must be 'random'")
    if str(training_cfg.get("framework") or "").strip().lower() != "jax":
        errors.append("training.framework must be 'jax'")
    if str(training_cfg.get("method_family") or "").strip().lower() != "rl":
        errors.append("training.method_family must be 'rl'")
    if str(training_cfg.get("algorithm") or "").strip().lower() not in {"ppo", "ppo_rnn"}:
        errors.append("training.algorithm must be 'ppo' or 'ppo_rnn'")

    try:
        max_params_million = float(model_cfg.get("max_params_million", 0))
    except (TypeError, ValueError):
        errors.append("model.max_params_million must be numeric")
    else:
        if max_params_million <= 0:
            errors.append("model.max_params_million must be > 0")
        if max_params_million > 100:
            errors.append("model.max_params_million must be <= 100")

    if int(training_cfg.get("env_step_budget", 0) or 0) <= 0:
        errors.append("training.env_step_budget must be > 0")
    if int(training_cfg.get("num_envs", 0) or 0) <= 0:
        errors.append("training.num_envs must be > 0")
    if int(training_cfg.get("num_steps", 0) or 0) <= 0:
        errors.append("training.num_steps must be > 0")
    if int(eval_cfg.get("num_episodes", 0) or 0) <= 0:
        errors.append("evaluation.num_episodes must be > 0")
    if int(eval_cfg.get("num_envs", 0) or 0) <= 0:
        errors.append("evaluation.num_envs must be > 0")
    if modal_cfg and not str(modal_cfg.get("gpu") or "").strip():
        errors.append("modal.gpu must be set when modal config is present")
    return errors


def build_summary(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    return {
        "timestamp_utc": now_utc_iso(),
        "track": TRACK_ID,
        "config_path": str(config_path),
        "task": config.get("task", {}),
        "environment": config.get("environment", {}),
        "model": config.get("model", {}),
        "training": config.get("training", {}),
        "evaluation": config.get("evaluation", {}),
        "modal": config.get("modal", {}),
        "hardware": config.get("hardware", {}),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "dependency_status": dependency_status(),
        "notes": [
            "Classic intentionally avoids the repo's container and Modal abstractions for environment interaction.",
            "The first in-repo baseline is PPO-RNN on Craftax-Classic symbolic observations.",
        ],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check_jax_ready() -> None:
    if JAX_IMPORT_ERROR is not None:
        raise RuntimeError(
            "classic baseline requires jax/craftax/flax/optax/distrax/orbax dependencies"
        ) from JAX_IMPORT_ERROR


def _upper_config(config: dict[str, Any]) -> dict[str, Any]:
    env_cfg = config.get("environment", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    return {
        "ENV_NAME": str(env_cfg.get("env_name", "Craftax-Classic-Symbolic-v1")),
        "USE_OPTIMISTIC_RESETS": bool(training_cfg.get("use_optimistic_resets", True)),
        "OPTIMISTIC_RESET_RATIO": int(training_cfg.get("optimistic_reset_ratio", 16)),
        "NUM_ENVS": int(training_cfg.get("num_envs", 1024)),
        "TOTAL_TIMESTEPS": int(training_cfg.get("env_step_budget", 1_000_000)),
        "LR": float(training_cfg.get("learning_rate", 2e-4)),
        "NUM_STEPS": int(training_cfg.get("num_steps", 64)),
        "UPDATE_EPOCHS": int(training_cfg.get("update_epochs", 4)),
        "NUM_MINIBATCHES": int(training_cfg.get("num_minibatches", 8)),
        "GAMMA": float(training_cfg.get("gamma", 0.99)),
        "GAE_LAMBDA": float(training_cfg.get("gae_lambda", 0.8)),
        "CLIP_EPS": float(training_cfg.get("clip_eps", 0.2)),
        "ENT_COEF": float(training_cfg.get("ent_coef", 0.01)),
        "VF_COEF": float(training_cfg.get("vf_coef", 0.5)),
        "MAX_GRAD_NORM": float(training_cfg.get("max_grad_norm", 1.0)),
        "ANNEAL_LR": bool(training_cfg.get("anneal_lr", True)),
        "SEED": int(training_cfg.get("seed", 0)),
        "LAYER_SIZE": int(model_cfg.get("layer_size", 256)),
        "EVAL_NUM_ENVS": int(eval_cfg.get("num_envs", 2048)),
        "EVAL_NUM_EPISODES": int(eval_cfg.get("num_episodes", 256)),
        "EVAL_GREEDY": bool(eval_cfg.get("greedy", True)),
        "EVAL_MAX_STEPS_PER_EPISODE": int(eval_cfg.get("max_steps_per_episode", 2000)),
        "EVAL_CHUNK_STEPS": int(eval_cfg.get("chunk_steps", 64)),
        "EVAL_SEED": int(eval_cfg.get("seed", 123)),
        "NUM_REPEATS": int(training_cfg.get("num_repeats", 1)),
    }


if JAX_IMPORT_ERROR is None:
    class GymnaxWrapper:
        def __init__(self, env: Any) -> None:
            self._env = env

        def __getattr__(self, name: str) -> Any:
            return getattr(self._env, name)


    class BatchEnvWrapper(GymnaxWrapper):
        def __init__(self, env: Any, num_envs: int) -> None:
            super().__init__(env)
            self.num_envs = num_envs
            self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
            self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

        @partial(jax.jit, static_argnums=(0, 2))
        def reset(self, rng: Any, params: Any = None) -> tuple[Any, Any]:
            rng, inner_rng = jax.random.split(rng)
            rngs = jax.random.split(inner_rng, self.num_envs)
            return self.reset_fn(rngs, params)

        @partial(jax.jit, static_argnums=(0, 4))
        def step(self, rng: Any, state: Any, action: Any, params: Any = None) -> tuple[Any, Any, Any, Any, Any]:
            rng, inner_rng = jax.random.split(rng)
            rngs = jax.random.split(inner_rng, self.num_envs)
            return self.step_fn(rngs, state, action, params)


    class AutoResetEnvWrapper(GymnaxWrapper):
        @partial(jax.jit, static_argnums=(0, 2))
        def reset(self, key: Any, params: Any = None) -> tuple[Any, Any]:
            return self._env.reset(key, params)

        @partial(jax.jit, static_argnums=(0, 4))
        def step(self, rng: Any, state: Any, action: Any, params: Any = None) -> tuple[Any, Any, Any, Any, Any]:
            rng, step_rng = jax.random.split(rng)
            obs_step, state_step, reward, done, info = self._env.step(step_rng, state, action, params)
            rng, reset_rng = jax.random.split(rng)
            obs_reset, state_reset = self._env.reset(reset_rng, params)

            def auto_reset(done_flag: Any, reset_state: Any, stepped_state: Any, reset_obs: Any, stepped_obs: Any) -> tuple[Any, Any]:
                state_out = jax.tree_util.tree_map(
                    lambda x, y: jax.lax.select(done_flag, x, y), reset_state, stepped_state
                )
                obs_out = jax.lax.select(done_flag, reset_obs, stepped_obs)
                return obs_out, state_out

            obs, state = auto_reset(done, state_reset, state_step, obs_reset, obs_step)
            return obs, state, reward, done, info


    class OptimisticResetVecEnvWrapper(GymnaxWrapper):
        def __init__(self, env: Any, num_envs: int, reset_ratio: int) -> None:
            super().__init__(env)
            self.num_envs = num_envs
            self.reset_ratio = reset_ratio
            if num_envs % reset_ratio != 0:
                raise ValueError("reset_ratio must divide num_envs")
            self.num_resets = self.num_envs // self.reset_ratio
            self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
            self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

        @partial(jax.jit, static_argnums=(0, 2))
        def reset(self, rng: Any, params: Any = None) -> tuple[Any, Any]:
            rng, inner_rng = jax.random.split(rng)
            rngs = jax.random.split(inner_rng, self.num_envs)
            return self.reset_fn(rngs, params)

        @partial(jax.jit, static_argnums=(0, 4))
        def step(self, rng: Any, state: Any, action: Any, params: Any = None) -> tuple[Any, Any, Any, Any, Any]:
            rng, step_rng = jax.random.split(rng)
            rngs = jax.random.split(step_rng, self.num_envs)
            obs_step, state_step, reward, done, info = self.step_fn(rngs, state, action, params)

            rng, reset_rng = jax.random.split(rng)
            reset_rngs = jax.random.split(reset_rng, self.num_resets)
            obs_reset, state_reset = self.reset_fn(reset_rngs, params)

            rng, choice_rng = jax.random.split(rng)
            reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)
            being_reset = jax.random.choice(
                choice_rng,
                jnp.arange(self.num_envs),
                shape=(self.num_resets,),
                p=done,
                replace=False,
            )
            reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))
            obs_reset = obs_reset[reset_indexes]
            state_reset = jax.tree_util.tree_map(lambda x: x[reset_indexes], state_reset)

            def auto_reset(done_flag: Any, reset_state: Any, stepped_state: Any, reset_obs: Any, stepped_obs: Any) -> tuple[Any, Any]:
                state_out = jax.tree_util.tree_map(
                    lambda x, y: jax.lax.select(done_flag, x, y), reset_state, stepped_state
                )
                obs_out = jax.lax.select(done_flag, reset_obs, stepped_obs)
                return state_out, obs_out

            state, obs = jax.vmap(auto_reset)(done, state_reset, state_step, obs_reset, obs_step)
            return obs, state, reward, done, info


    @struct.dataclass
    class LogEnvState:
        env_state: Any
        episode_returns: float
        episode_lengths: int
        returned_episode_returns: float
        returned_episode_lengths: int
        timestep: int


    class LogWrapper(GymnaxWrapper):
        @partial(jax.jit, static_argnums=(0, 2))
        def reset(self, key: chex.PRNGKey, params: Any = None) -> tuple[Any, Any]:
            obs, env_state = self._env.reset(key, params)
            state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
            return obs, state

        @partial(jax.jit, static_argnums=(0, 4))
        def step(self, key: chex.PRNGKey, state: Any, action: Any, params: Any = None) -> tuple[Any, Any, Any, Any, Any]:
            obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
            new_episode_return = state.episode_returns + reward
            new_episode_length = state.episode_lengths + 1
            state = LogEnvState(
                env_state=env_state,
                episode_returns=new_episode_return * (1 - done),
                episode_lengths=new_episode_length * (1 - done),
                returned_episode_returns=state.returned_episode_returns * (1 - done) + new_episode_return * done,
                returned_episode_lengths=state.returned_episode_lengths * (1 - done) + new_episode_length * done,
                timestep=state.timestep + 1,
            )
            info["returned_episode_returns"] = state.returned_episode_returns
            info["returned_episode_lengths"] = state.returned_episode_lengths
            info["timestep"] = state.timestep
            info["returned_episode"] = done
            return obs, state, reward, done, info


    class ScannedRNN(nn.Module):
        @partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
        @nn.compact
        def __call__(self, carry: Any, x: tuple[Any, Any]) -> tuple[Any, Any]:
            resets = x[1]
            inputs = x[0]
            carry = jnp.where(
                resets[:, np.newaxis],
                self.initialize_carry(inputs.shape[0], inputs.shape[1]),
                carry,
            )
            new_carry, y = nn.GRUCell(features=inputs.shape[1])(carry, inputs)
            return new_carry, y

        @staticmethod
        def initialize_carry(batch_size: int, hidden_size: int) -> Any:
            cell = nn.GRUCell(features=hidden_size)
            return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


    class ActorCriticRNN(nn.Module):
        action_dim: int
        config: dict[str, Any]

        @nn.compact
        def __call__(self, hidden: Any, x: tuple[Any, Any]) -> tuple[Any, Any, Any]:
            obs, dones = x
            embedding = nn.Dense(
                self.config["LAYER_SIZE"],
                kernel_init=orthogonal(math.sqrt(2)),
                bias_init=constant(0.0),
            )(obs)
            embedding = nn.relu(embedding)
            hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

            actor_mean = nn.Dense(self.config["LAYER_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
            actor_mean = nn.relu(actor_mean)
            actor_mean = nn.Dense(self.config["LAYER_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(actor_mean)
            actor_mean = nn.relu(actor_mean)
            actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
            pi = distrax.Categorical(logits=actor_mean)

            critic = nn.Dense(self.config["LAYER_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
            critic = nn.relu(critic)
            critic = nn.Dense(self.config["LAYER_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(critic)
            critic = nn.relu(critic)
            critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
            return hidden, pi, jnp.squeeze(critic, axis=-1)


    class Transition(NamedTuple):
        done: Any
        action: Any
        value: Any
        reward: Any
        log_prob: Any
        obs: Any
        info: Any


    @dataclass
    class TrainArtifacts:
        train_state: Any
        metrics: dict[str, Any]
        output_dir: Path


    def _build_env(env_name: str, num_envs: int, use_optimistic_resets: bool, optimistic_reset_ratio: int) -> tuple[Any, Any]:
        env = make_craftax_env_from_name(env_name, not use_optimistic_resets)
        env_params = env.default_params
        env = LogWrapper(env)
        if use_optimistic_resets:
            env = OptimisticResetVecEnvWrapper(
                env,
                num_envs=num_envs,
                reset_ratio=min(optimistic_reset_ratio, num_envs),
            )
        else:
            env = AutoResetEnvWrapper(env)
            env = BatchEnvWrapper(env, num_envs=num_envs)
        return env, env_params


    def make_train(config: dict[str, Any]) -> Any:
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
        env, env_params = _build_env(
            env_name=config["ENV_NAME"],
            num_envs=config["NUM_ENVS"],
            use_optimistic_resets=config["USE_OPTIMISTIC_RESETS"],
            optimistic_reset_ratio=config["OPTIMISTIC_RESET_RATIO"],
        )

        def linear_schedule(count: Any) -> Any:
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng: Any) -> dict[str, Any]:
            network = ActorCriticRNN(env.action_space(env_params).n, config=config)
            rng, init_rng = jax.random.split(rng)
            init_obs = (
                jnp.zeros((1, config["NUM_ENVS"], *env.observation_space(env_params).shape)),
                jnp.zeros((1, config["NUM_ENVS"])),
            )
            init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["LAYER_SIZE"])
            network_params = network.init(init_rng, init_hstate, init_obs)
            if config["ANNEAL_LR"]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5),
                )
            train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

            rng, reset_rng = jax.random.split(rng)
            obsv, env_state = env.reset(reset_rng, env_params)
            init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["LAYER_SIZE"])

            def _update_step(runner_state: Any, _: Any) -> tuple[Any, Any]:
                def _env_step(inner_state: Any, _: Any) -> tuple[Any, Any]:
                    train_state, env_state, last_obs, last_done, hstate, rng, update_step = inner_state
                    rng, action_rng = jax.random.split(rng)
                    ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                    hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                    action = pi.sample(seed=action_rng)
                    log_prob = pi.log_prob(action)
                    value = value.squeeze(0)
                    action = action.squeeze(0)
                    log_prob = log_prob.squeeze(0)

                    rng, env_rng = jax.random.split(rng)
                    obsv, env_state, reward, done, info = env.step(env_rng, env_state, action, env_params)
                    transition = Transition(last_done, action, value, reward, log_prob, last_obs, info)
                    next_state = (train_state, env_state, obsv, done, hstate, rng, update_step)
                    return next_state, transition

                initial_hstate = runner_state[-3]
                runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])
                train_state, env_state, last_obs, last_done, hstate, rng, update_step = runner_state
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                _, _, last_val = network.apply(train_state.params, hstate, ac_in)
                last_val = last_val.squeeze(0)

                def _calculate_gae(traj: Any, bootstrap_val: Any, bootstrap_done: Any) -> tuple[Any, Any]:
                    def _gae_step(carry: Any, transition: Transition) -> tuple[Any, Any]:
                        gae, next_value, next_done = carry
                        delta = transition.reward + config["GAMMA"] * next_value * (1 - next_done) - transition.value
                        gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                        return (gae, transition.value, transition.done), gae

                    _, advantages = jax.lax.scan(
                        _gae_step,
                        (jnp.zeros_like(bootstrap_val), bootstrap_val, bootstrap_done),
                        traj,
                        reverse=True,
                        unroll=16,
                    )
                    return advantages, advantages + traj.value

                advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

                def _update_epoch(update_state: Any, _: Any) -> tuple[Any, Any]:
                    def _update_minibatch(train_state: Any, batch_info: Any) -> tuple[Any, Any]:
                        init_hstate, traj_batch, advantages, targets = batch_info

                        def _loss_fn(params: Any, init_hstate: Any, traj_batch: Any, gae: Any, targets: Any) -> tuple[Any, Any]:
                            _, pi, value = network.apply(params, init_hstate[0], (traj_batch.obs, traj_batch.done))
                            log_prob = pi.log_prob(traj_batch.action)
                            value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                                -config["CLIP_EPS"], config["CLIP_EPS"]
                            )
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                            ratio = jnp.exp(log_prob - traj_batch.log_prob)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                            entropy = pi.entropy().mean()
                            total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                            return total_loss, (value_loss, loss_actor, entropy)

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                        train_state = train_state.apply_gradients(grads=grads)
                        return train_state, total_loss

                    train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                    rng, perm_rng = jax.random.split(rng)
                    permutation = jax.random.permutation(perm_rng, config["NUM_ENVS"])
                    batch = (init_hstate, traj_batch, advantages, targets)
                    shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)
                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                            ),
                            1,
                            0,
                        ),
                        shuffled_batch,
                    )
                    train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)
                    return (train_state, init_hstate, traj_batch, advantages, targets, rng), total_loss

                update_state = (
                    train_state,
                    initial_hstate[None, :],
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state = update_state[0]
                metric = jax.tree_util.tree_map(
                    lambda x: (x * traj_batch.info["returned_episode"]).sum() / jnp.maximum(traj_batch.info["returned_episode"].sum(), 1),
                    traj_batch.info,
                )
                runner_state = (train_state, env_state, last_obs, last_done, hstate, update_state[-1], update_step + 1)
                return runner_state, {"episode_metrics": metric, "loss_info": loss_info}

            rng, loop_rng = jax.random.split(rng)
            runner_state = (
                train_state,
                env_state,
                obsv,
                jnp.zeros((config["NUM_ENVS"]), dtype=bool),
                init_hstate,
                loop_rng,
                0,
            )
            runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
            return {"runner_state": runner_state, "metrics": metrics}

        return train


    def _save_train_state(train_state: Any, output_dir: Path, total_timesteps: int) -> Path:
        checkpoint_root = output_dir / "checkpoints"
        checkpointer = PyTreeCheckpointer()
        options = CheckpointManagerOptions(max_to_keep=1, create=True)
        manager = CheckpointManager(checkpoint_root.as_posix(), checkpointer, options)
        save_args = orbax_utils.save_args_from_target(train_state)
        manager.save(total_timesteps, train_state, save_kwargs={"save_args": save_args})
        return checkpoint_root


    def _restore_train_state(config: dict[str, Any], output_dir: Path) -> Any:
        upper = _upper_config(config)
        env = make_craftax_env_from_name(upper["ENV_NAME"], True)
        env_params = env.default_params
        network = ActorCriticRNN(env.action_space(env_params).n, config=upper)
        init_obs = (
            jnp.zeros((1, 1, *env.observation_space(env_params).shape)),
            jnp.zeros((1, 1)),
        )
        init_hstate = ScannedRNN.initialize_carry(1, upper["LAYER_SIZE"])
        network_params = network.init(jax.random.PRNGKey(0), init_hstate, init_obs)
        tx = optax.chain(
            optax.clip_by_global_norm(upper["MAX_GRAD_NORM"]),
            optax.adam(upper["LR"], eps=1e-5),
        )
        template = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
        checkpoint_root = output_dir / "checkpoints"
        manager = CheckpointManager(
            checkpoint_root.as_posix(),
            PyTreeCheckpointer(),
            CheckpointManagerOptions(max_to_keep=1, create=True),
        )
        return manager.restore(upper["TOTAL_TIMESTEPS"], items=template)


    def evaluate_policy(config: dict[str, Any], train_state: Any) -> dict[str, Any]:
        upper = _upper_config(config)
        print(
            json.dumps(
                {
                    "stage": "eval_start",
                    "num_envs": upper["EVAL_NUM_ENVS"],
                    "num_episodes": upper["EVAL_NUM_EPISODES"],
                    "chunk_steps": upper["EVAL_CHUNK_STEPS"],
                    "greedy": upper["EVAL_GREEDY"],
                }
            ),
            flush=True,
        )
        env, env_params = _build_env(
            env_name=upper["ENV_NAME"],
            num_envs=upper["EVAL_NUM_ENVS"],
            use_optimistic_resets=False,
            optimistic_reset_ratio=1,
        )
        network = ActorCriticRNN(env.action_space(env_params).n, config=upper)
        rng = jax.random.PRNGKey(upper["EVAL_SEED"])
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng, env_params)
        hstate = ScannedRNN.initialize_carry(upper["EVAL_NUM_ENVS"], upper["LAYER_SIZE"])
        done = jnp.zeros((upper["EVAL_NUM_ENVS"],), dtype=bool)

        @partial(jax.jit, static_argnums=(5,))
        def rollout_chunk(
            env_state: Any,
            obs: Any,
            done: Any,
            hstate: Any,
            rng: Any,
            num_steps: int,
        ) -> tuple[Any, Any]:
            def _step(carry: Any, _: Any) -> tuple[Any, Any]:
                env_state_t, obs_t, done_t, hstate_t, rng_t = carry
                rng_t, action_rng, env_rng = jax.random.split(rng_t, 3)
                ac_in = (obs_t[np.newaxis, :], done_t[np.newaxis, :])
                hstate_t, pi, _ = network.apply(train_state.params, hstate_t, ac_in)
                action = pi.mode() if upper["EVAL_GREEDY"] else pi.sample(seed=action_rng)
                action = action.squeeze(0)
                next_obs, next_env_state, _reward, next_done, info = env.step(env_rng, env_state_t, action, env_params)
                return (next_env_state, next_obs, next_done, hstate_t, rng_t), info

            return jax.lax.scan(_step, (env_state, obs, done, hstate, rng), None, length=num_steps)

        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        max_total_steps = upper["EVAL_NUM_EPISODES"] * upper["EVAL_MAX_STEPS_PER_EPISODE"]
        total_env_steps = 0
        chunk_count = 0

        while len(episode_returns) < upper["EVAL_NUM_EPISODES"] and total_env_steps < max_total_steps:
            (env_state, obs, done, hstate, rng), info = rollout_chunk(
                env_state,
                obs,
                done,
                hstate,
                rng,
                upper["EVAL_CHUNK_STEPS"],
            )
            chunk_count += 1
            returned = np.asarray(info["returned_episode"]).astype(bool)
            returns = np.asarray(info["returned_episode_returns"], dtype=float)
            lengths = np.asarray(info["returned_episode_lengths"], dtype=float)
            if returned.any():
                episode_returns.extend(returns[returned].tolist())
                episode_lengths.extend(lengths[returned].astype(int).tolist())
            total_env_steps += upper["EVAL_NUM_ENVS"] * upper["EVAL_CHUNK_STEPS"]
            if chunk_count == 1 or chunk_count % 10 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "eval_progress",
                            "chunk_count": chunk_count,
                            "episodes_collected": len(episode_returns),
                            "total_env_steps": total_env_steps,
                        }
                    ),
                    flush=True,
                )

        if not episode_returns:
            raise RuntimeError("evaluation completed without any finished episodes")

        returns_array = np.asarray(episode_returns[: upper["EVAL_NUM_EPISODES"]], dtype=float)
        lengths_array = np.asarray(episode_lengths[: upper["EVAL_NUM_EPISODES"]], dtype=float)
        return {
            "num_eval_episodes": int(len(returns_array)),
            "mean_episode_return": float(returns_array.mean()),
            "std_episode_return": float(returns_array.std()),
            "mean_episode_length": float(lengths_array.mean()),
            "median_episode_return": float(np.median(returns_array)),
            "greedy": bool(upper["EVAL_GREEDY"]),
            "num_eval_envs": int(upper["EVAL_NUM_ENVS"]),
            "eval_seed": int(upper["EVAL_SEED"]),
        }


    def run_training(config: dict[str, Any], output_dir: Path) -> TrainArtifacts:
        upper = _upper_config(config)
        print(
            json.dumps(
                {
                    "stage": "train_start",
                    "env_name": upper["ENV_NAME"],
                    "num_envs": upper["NUM_ENVS"],
                    "total_timesteps": upper["TOTAL_TIMESTEPS"],
                }
            ),
            flush=True,
        )
        train_fn = jax.jit(make_train(upper))
        rng = jax.random.PRNGKey(upper["SEED"])
        started = time.time()
        result = train_fn(rng)
        result = jax.block_until_ready(result)
        duration = time.time() - started
        runner_state = result["runner_state"]
        train_state = runner_state[0]
        checkpoint_root = _save_train_state(train_state, output_dir, upper["TOTAL_TIMESTEPS"])
        print(
            json.dumps(
                {
                    "stage": "train_saved_checkpoint",
                    "checkpoint_root": checkpoint_root.as_posix(),
                    "training_seconds": duration,
                }
            ),
            flush=True,
        )
        metrics = {
            "track": TRACK_ID,
            "algorithm": config.get("training", {}).get("algorithm", "ppo_rnn"),
            "env_name": upper["ENV_NAME"],
            "total_timesteps": int(upper["TOTAL_TIMESTEPS"]),
            "num_envs": int(upper["NUM_ENVS"]),
            "num_steps": int(upper["NUM_STEPS"]),
            "num_updates": int(upper["NUM_UPDATES"]),
            "layer_size": int(upper["LAYER_SIZE"]),
            "training_seconds": float(duration),
            "steps_per_second": float(upper["TOTAL_TIMESTEPS"] / max(duration, 1e-6)),
            "checkpoint_root": checkpoint_root.as_posix(),
        }
        return TrainArtifacts(train_state=train_state, metrics=metrics, output_dir=output_dir)


else:  # pragma: no cover - used only for dependency-free preflight paths
    TrainArtifacts = Any  # type: ignore[assignment]


def _parameter_count(train_state: Any) -> int | None:
    if JAX_IMPORT_ERROR is not None:
        return None
    leaves = jax.tree_util.tree_leaves(train_state.params)
    return int(sum(int(leaf.size) for leaf in leaves))


def _write_run_outputs(
    config: dict[str, Any],
    output_dir: Path,
    summary: dict[str, Any],
    train_artifacts: Any | None = None,
    eval_summary: dict[str, Any] | None = None,
) -> None:
    _write_json(output_dir / "classic_preflight.json", summary)
    if train_artifacts is not None:
        metrics = dict(train_artifacts.metrics)
        param_count = _parameter_count(train_artifacts.train_state)
        if param_count is not None:
            metrics["parameter_count"] = param_count
            metrics["parameter_count_million"] = float(param_count / 1_000_000)
        _write_json(output_dir / "metrics.json", metrics)
    if eval_summary is not None:
        _write_json(output_dir / "eval_summary.json", eval_summary)
    (output_dir / "run_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_dir / "command.txt").write_text("python -m nanohorizon.baselines.classic\n", encoding="utf-8")


def _write_training_outputs(
    config: dict[str, Any],
    output_dir: Path,
    summary: dict[str, Any],
    train_artifacts: Any,
) -> None:
    _write_run_outputs(config, output_dir, summary, train_artifacts=train_artifacts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Craftax-Classic PPO baseline and eval runner.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the classic track config YAML.")
    parser.add_argument("--output-dir", default="", help="Optional directory for run outputs.")
    parser.add_argument(
        "--skip-dependency-check",
        action="store_true",
        help="Skip failing when jax/craftax are not installed.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the PPO baseline and checkpoint save path.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load a saved checkpoint from output-dir and run parallel eval only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    errors = validate_config(config)
    summary = build_summary(config, config_path)
    summary["config_validation_errors"] = errors

    output_dir_raw = str(args.output_dir or config.get("output", {}).get("root_dir") or "").strip()
    output_dir = ensure_dir(output_dir_raw) if output_dir_raw else None

    if output_dir is not None:
        _write_json(output_dir / "classic_preflight.json", summary)

    print(json.dumps(summary, indent=2, sort_keys=True))

    if errors:
        raise SystemExit("classic config validation failed")
    if not args.skip_dependency_check and not summary["dependency_status"]["ready"]:
        raise SystemExit(
            "classic dependency check failed; install the JAX/Craftax stack or rerun with --skip-dependency-check"
        )
    if not args.train and not args.eval_only:
        return
    if output_dir is None:
        raise SystemExit("training/eval requires --output-dir or output.root_dir in config")

    _check_jax_ready()
    if args.eval_only:
        print(json.dumps({"stage": "restore_start"}, sort_keys=True), flush=True)
        train_state = _restore_train_state(config, output_dir)
        print(json.dumps({"stage": "restore_done"}, sort_keys=True), flush=True)
        eval_summary = evaluate_policy(config, train_state)
        _write_run_outputs(config, output_dir, summary, eval_summary=eval_summary)
        print(json.dumps(eval_summary, indent=2, sort_keys=True))
        return

    train_artifacts = run_training(config, output_dir)
    _write_training_outputs(config, output_dir, summary, train_artifacts)
    eval_summary = evaluate_policy(config, train_artifacts.train_state)
    _write_run_outputs(config, output_dir, summary, train_artifacts=train_artifacts, eval_summary=eval_summary)
    print(json.dumps(train_artifacts.metrics, indent=2, sort_keys=True))
    print(json.dumps(eval_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
