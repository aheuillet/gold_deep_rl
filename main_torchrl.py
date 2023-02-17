import torch
import tqdm #replace with rich
import gym
from functorch import vmap
from matplotlib import pyplot as plt
from tensordict import TensorDict
from tensordict.nn import get_functional
from torch import nn
from tensordict.tensordict import pad
from torchrl.objectives.value.functional import vec_td_lambda_advantage_estimate
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    CatFrames,
    CatTensors,
    Compose,
    GrayScale,
    ObservationNorm,
    Resize,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.utils import set_exploration_mode, step_mdp
from torchrl.modules import DuelingCnnDQNet, EGreedyWrapper, QValueActor
from CustomPyBoyGym import CustomPyBoyGym
from pyboy.pyboy import *
from pathlib import Path
import sys
# hyperparams

# the learning rate of the optimizer
lr = 2e-3
# the beta parameters of Adam
betas = (0.9, 0.999)
# gamma decay factor
gamma = 0.99
# lambda decay factor (see second the part with TD(lambda)
lmbda = 0.95
# total frames collected in the environment. In other implementations, the user defines a maximum number of episodes.
# This is harder to do with our data collectors since they return batches of N collected frames, where N is a constant.
# However, one can easily get the same restriction on number of episodes by breaking the training loop when a certain number
# episodes has been collected.
total_frames = 500000
# Random frames used to initialize the replay buffer.
init_random_frames = 100
# Frames in each batch collected.
frames_per_batch = 32
# Optimization steps per batch collected
n_optim = 4
# Frames sampled from the replay buffer at each optimization step
batch_size = 32
# Size of the replay buffer in terms of frames
buffer_size = min(total_frames, 100000)
# Number of environments run in parallel in each data collector
n_workers = 1

device = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"

# Smooth target network update decay parameter. This loosely corresponds to a 1/(1-tau) interval with hard target network update
tau = 0.005

# Initial and final value of the epsilon factor in Epsilon-greedy exploration (notice that since our policy is deterministic exploration is crucial)
eps_greedy_val = 0.1
eps_greedy_val_env = 0.05

# To speed up learning, we set the bias of the last layer of our value network to a predefined value
init_bias = 20.0

def make_env(game, mode="headless", parallel=False):

    f = lambda: PyBoy(game, window_type=mode, window_scale=3, debug=False, game_wrapper=True)

    if parallel:
        base_env = ParallelEnv(
            n_workers,
            EnvCreator(
                lambda: GymEnv(
                    "PyBoy", pyboy=f(), from_pixels=True, pixels_only=True, device=device
                )
            ),
        )
    else:
        base_env = GymEnv(
            "PyBoy", pyboy=f(), from_pixels=True, pixels_only=True, device=device
        )

    env = TransformedEnv(
        base_env,
        Compose(
            ToTensorImage(),
            Resize(80, 80),
            ObservationNorm(in_keys=["pixels"], standard_normal=True),
            CatFrames(4, in_keys=["pixels"], dim=-3),
        ),
    )
    env.transform.init_stats(100)
    return env

def make_model(dummy_env):
    cnn_kwargs = {
        "in_features": 3,
        "num_cells": [32, 64, 64],
        "kernel_sizes": [6, 4, 3],
        "strides": [2, 2, 1],
        "activation_class": nn.ELU,
        "squeeze_output": True,
        "aggregator_class": nn.AdaptiveAvgPool2d,
        "aggregator_kwargs": {"output_size": (1, 1)},
    }
    mlp_kwargs = {
        "depth": 2,
        "num_cells": [
            64,
            64,
        ],
        # "out_features": dummy_env.action_spec.shape[-1],
        "activation_class": nn.ELU,
    }
    
    net = DuelingCnnDQNet(
        dummy_env.action_spec.shape[-1], 1, cnn_kwargs, mlp_kwargs
    ).to(device)
    net.value[-1].bias.data.fill_(init_bias)

    actor = QValueActor(net, in_keys=["pixels"], spec=dummy_env.action_spec).to(device)
    # init actor
    tensordict = dummy_env.reset()
    print("reset results:", tensordict)
    actor(tensordict)
    print("Q-value network results:", tensordict)

    # make functional
    # here's an explicit way of creating the parameters and buffer tensordict.
    # Alternatively, we could have used `params = make_functional(actor)` from
    # tensordict.nn
    params = TensorDict({k: v for k, v in actor.named_parameters()}, [])
    buffers = TensorDict({k: v for k, v in actor.named_buffers()}, [])
    params = params.update(buffers).unflatten_keys(".")  # creates a nested TensorDict
    factor = get_functional(actor)

    # creating the target parameters is fairly easy with tensordict:
    (params_target,) = (params.to_tensordict().detach(),)

    # we wrap our actor in an EGreedyWrapper for data collection
    actor_explore = EGreedyWrapper(
        actor,
        annealing_num_steps=total_frames,
        eps_init=eps_greedy_val,
        eps_end=eps_greedy_val_env,
    )

    return factor, actor, actor_explore, params, params_target    


if __name__ == "main":

    """
    Choose game
    """
    gamesFolder = Path("games")
    games = [os.path.join(gamesFolder, f) for f in os.listdir(gamesFolder) if (os.path.isfile(os.path.join(gamesFolder, f)) and f.endswith(".gbc"))]
    gameNames = [f.replace(".gbc", "") for f in os.listdir(gamesFolder) if (os.path.isfile(os.path.join(gamesFolder, f)) and f.endswith(".gbc"))]

    print("Avaliable games: ", games)
    for cnt, gameName in enumerate(games, 1):
        sys.stdout.write("[%d] %s\n\r" % (cnt, gameName))

    choice = int(input("Select game[1-%s]: " % cnt)) - 1
    game = games[choice]
    gameName = gameNames[choice]


    dummy_env = make_env(parallel=False)
    (
    factor,
    actor,
    actor_explore,
    params,
    params_target,
    ) = make_model(dummy_env)
    params_flat = params.flatten_keys(".")
    params_target_flat = params_target.flatten_keys(".")
    #register gym env
    gym.register("PyBoy", entry_point=CustomPyBoyGym, kwargs={"observation_type": "raw"})

    max_size = frames_per_batch // n_workers

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(-(-buffer_size // max_size)),
        prefetch=n_optim,
    )

    data_collector = MultiaSyncDataCollector(
        [make_env(game, parallel=True), make_env(game, parallel=True)],
        policy=actor_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        exploration_mode="random",
        devices=[device, device],
        storing_devices=[device, device],
    )
    
    optim = torch.optim.Adam(list(params_flat.values()), lr)
    #print(actor_explore(dummy_env.reset()))

    evals = []
    traj_lengths_eval = []
    losses = []
    frames = []
    values = []
    grad_vals = []
    traj_lengths = []
    mavgs = []
    traj_count = []
    prev_traj_count = 0

    pbar = tqdm.tqdm(total=total_frames)
    for j, data in enumerate(data_collector):
        mask = data["collector", "mask"]
        data = pad(data, [0, 0, 0, max_size - data.shape[1]])
        current_frames = mask.sum().cpu().item()
        pbar.update(current_frames)

        replay_buffer.extend(data.cpu())
        if len(frames):
            frames.append(current_frames + frames[-1])
        else:
            frames.append(current_frames)

        if data["done"].any():
            done = data["done"].squeeze(-1)
            traj_lengths.append(data["collector", "step_count"][done].float().mean().item())

        if sum(frames) > init_random_frames:
            for _ in range(n_optim):
                sampled_data = replay_buffer.sample(batch_size // max_size)
                sampled_data = sampled_data.clone().to(device, non_blocking=True)

                reward = sampled_data["reward"]
                done = sampled_data["done"].to(reward.dtype)
                action = sampled_data["action"].clone()

                sampled_data_out = sampled_data.select(*actor.in_keys)
                sampled_data_out = vmap(factor, (0, None))(sampled_data_out, params)
                action_value = sampled_data_out["action_value"]
                action_value = (action_value * action.to(action_value.dtype)).sum(-1, True)
                with torch.no_grad():
                    tdstep = step_mdp(sampled_data)
                    next_value = vmap(factor, (0, None))(
                        tdstep.select(*actor.in_keys), params
                    )
                    next_value = next_value["chosen_action_value"]
                error = vec_td_lambda_advantage_estimate(
                    gamma,
                    lmbda,
                    action_value,
                    next_value,
                    reward,
                    done,
                ).pow(2)
                # reward + gamma * next_value * (1 - done)
                mask = sampled_data["collector", "mask"]
                error = error[mask].mean()
                # assert exp_value.shape == action_value.shape
                # error = nn.functional.smooth_l1_loss(exp_value, action_value).mean()
                # error = nn.functional.mse_loss(exp_value, action_value)[mask].mean()
                error.backward()

                # gv = sum([p.grad.pow(2).sum() for p in params_flat.values()]).sqrt()
                # nn.utils.clip_grad_value_(list(params_flat.values()), 1)
                gv = nn.utils.clip_grad_norm_(list(params_flat.values()), 100)

                optim.step()
                optim.zero_grad()

                for (key, p1) in params_flat.items():
                    p2 = params_target_flat[key]
                    params_target_flat.set_(key, tau * p1.data + (1 - tau) * p2.data)

            pbar.set_description(
                f"error: {error: 4.4f}, value: {action_value.mean(): 4.4f}"
            )
            actor_explore.step(current_frames)

            # logs
            with set_exploration_mode("random"), torch.inference_mode():
                #         eval_rollout = dummy_env.rollout(max_steps=1000, policy=actor_explore, auto_reset=True).cpu()
                eval_rollout = dummy_env.rollout(
                    max_steps=10000, policy=actor, auto_reset=True
                ).cpu()
            grad_vals.append(float(gv))
            traj_lengths_eval.append(eval_rollout.shape[-1])
            evals.append(eval_rollout["reward"].squeeze(-1).sum(-1).item())
            if len(mavgs):
                mavgs.append(evals[-1] * 0.05 + mavgs[-1] * 0.95)
            else:
                mavgs.append(evals[-1])
            losses.append(error.item())
            values.append(action_value[mask].mean().item())
            traj_count.append(prev_traj_count + data["done"].sum().item())
            prev_traj_count = traj_count[-1]
            # plots
            if j % 10 == 0:
                plt.clf()
                plt.figure(figsize=(15, 15))
                plt.subplot(3, 2, 1)
                plt.plot(frames[-len(evals) :], evals, label="return")
                plt.plot(frames[-len(mavgs) :], mavgs, label="mavg")
                plt.xlabel("frames collected")
                plt.ylabel("trajectory length (= return)")
                plt.subplot(3, 2, 2)
                plt.plot(traj_count[-len(evals) :], evals, label="return")
                plt.plot(traj_count[-len(mavgs) :], mavgs, label="mavg")
                plt.xlabel("trajectories collected")
                plt.legend()
                plt.subplot(3, 2, 3)
                plt.plot(frames[-len(losses) :], losses)
                plt.xlabel("frames collected")
                plt.title("loss")
                plt.subplot(3, 2, 4)
                plt.plot(frames[-len(values) :], values)
                plt.xlabel("frames collected")
                plt.title("value")
                plt.subplot(3, 2, 5)
                plt.plot(frames[-len(grad_vals) :], grad_vals)
                plt.xlabel("frames collected")
                plt.title("grad norm")
                if len(traj_lengths):
                    plt.subplot(3, 2, 6)
                    plt.plot(traj_lengths)
                    plt.xlabel("batches")
                    plt.title("traj length (training)")
            plt.savefig("dqn_tdlambda.png")

        # update policy weights
        data_collector.update_policy_weights_()

    print("shutting down")
    data_collector.shutdown()
    del data_collector

    # save results
    torch.save(
        {
            "frames": frames,
            "evals": evals,
            "mavgs": mavgs,
            "losses": losses,
            "values": values,
            "grad_vals": grad_vals,
            "traj_lengths_training": traj_lengths,
            "traj_count": traj_count,
            "weights": (params,),
        },
        "saved_results_tdlambda.pt",
    )