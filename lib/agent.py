import os
import math
import numpy as np
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .env.hiv_dynamics import HIV
from .model import A2C_CONTI_MODEL, A2C_DISCRETE_MODEL
from .utils import load_ckpt, save_ckpt


class A2C_AGENT:
    def __init__(self):
        pass

    def calc_logprob(self, mu, var, actions):
        p1 = -((mu - actions) ** 2) / (2 * var.clamp(min=1e-3))
        p2 = -(var.shape[1] / 2) * torch.log(2 * math.pi * var)
        return p1 + p2

    def choose_action(
        self, mu: torch.Tensor, var: torch.Tensor, stochastic: bool = True,
    ) -> np.ndarray:
        mu = mu.data.cpu().numpy()
        if stochastic:
            sigma = torch.sqrt(var).data.cpu().numpy()
            action = np.random.normal(mu, sigma)
        else:
            action = mu
        return action

    def train(self, config, logdir, resume=True):
        
        tb_dir = os.path.join(logdir, 'tensorboard')
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)

        device = config.device
        net = A2C_CONTI_MODEL(config, state_size=6, action_size=2).to(device)
        if config.multi_gpu:
            net = nn.DataParallel(net)
        optimizer = optim.Adam(net.parameters(), lr=config.train.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.max_episode)

        info = {
            'net': net,
            'optim': optimizer,
            'sched': scheduler,
            'episode': 0,
        }
        ckpt_dir = os.path.join(logdir, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)
        if resume:
            info = load_ckpt(ckpt_dir, info, device, train=True)

        start_episode = info['episode']
        max_episode = config.train.max_episode

        env = HIV(config)
        start_time = datetime.now()
        for episode in range(start_episode, max_episode):
            info['net'].train()
            env.reset()
            states = []
            actions = []
            ref_values = []

            # One trajectory
            for t in range(config.env.T_max):
                state = torch.from_numpy(env.state).type(torch.float32).to(device)
                states.append(state)
                mu, var, _ = info['net'](state)
                action = self.choose_action(mu, var)
                action = env.trim_action(action)
                actions.append(torch.from_numpy(action))

                next_state, reward, done = env.step(action)
                print(t, torch.cat([mu, var], dim=1)[0].data, int(next_state[0, -1]))
                next_state = torch.from_numpy(next_state).type(torch.float32).to(device)
                reward = torch.from_numpy(reward).type(torch.float32).to(device)
                if not done:
                    last_value = info['net'](next_state)[2]
                    reward += last_value.squeeze(-1)
                ref_values.append(reward)

            states = torch.cat(states, dim=0).to(device)
            actions = torch.cat(actions, dim=0).to(device)
            ref_values = torch.cat(ref_values, dim=0)

            info['optim'].zero_grad()
            mu, var, value = info['net'](states)
            
            loss_value = F.mse_loss(value.squeeze(-1), ref_values)

            advantage = ref_values.unsqueeze(-1) - value.detach()
            log_prob = advantage * self.calc_logprob(mu, var, actions)
            loss_policy = -log_prob.mean()

            entropy = -(torch.log(2 * math.pi * var) + 1) / 2
            loss_entropy = config.train.entropy_beta * entropy.mean()

            loss = loss_value + loss_policy + loss_entropy
            loss.backward()
            info['optim'].step()
            info['sched'].step()

            # Logging
            if info['episode'] % config.train.log_freq == 0 or info['episode'] == config.train.max_episode:
                logging.info(
                    f'epi {info["episode"]} total {loss.item():.3e} v {loss_value.item():.3e} '\
                    f'p {loss_policy.item():.3e} e {loss_entropy.item():.3e} '\
                    f'elapsed {datetime.now() - start_time}'
                )
                writer.add_scalar('total_loss', loss, info['episode'])
                writer.add_scalar('loss_value', loss_value, info['episode'])
                writer.add_scalar('loss_policy', loss_policy, info['episode'])
                writer.add_scalar('loss_entropy', loss_entropy, info['episode'])

            # Save checkpoint (save as ckpt.pt - overwritten)
            if info['episode'] % config.train.save_freq == 0 or info['episode'] == config.train.max_episode:
                save_ckpt(ckpt_dir, info, archive=False)

            # Evaluate
            if info['episode'] % config.train.eval_freq == 0:
                self.eval(config, logdir, None, writer)

            # Archive checkpoint (save as ckpt_123.pt)
            if info['episode'] % config.train.archive_freq == 0 or info['episode'] == config.train.max_episode:
                save_ckpt(ckpt_dir, info, archive=True)

            info['episode'] = episode + 1

    def eval(self, config, logdir, ckpt_num=None, writer=None):
        device = config.device
        net = A2C_CONTI_MODEL(config, state_size=6, action_size=2).to(device)
        if config.multi_gpu:
            net = nn.DataParallel(net)
        env = HIV(config)
        info = {
            'net': net,
            'episode': 0,
        }
        ckpt_dir = os.path.join(logdir, 'ckpt')
        info = load_ckpt(ckpt_dir, info, device, train=False, ckpt_num=ckpt_num)
        info['net'].eval()

        env = HIV(config, batch_size=1)
        states = []
        actions = []
        rewards = []

        for t in range(config.env.T_max):
            with torch.no_grad():
                state = torch.from_numpy(env.state).type(torch.float32).to(device)
                states.append(state)
                mu, var, _ = info['net'](state)
                action = self.choose_action(mu, var, stochastic=False)
                action = env.trim_action(action)
                actions.append(torch.from_numpy(action))
                _, reward, _ = env.step(action)
                rewards.append(torch.from_numpy(reward).unsqueeze(1))

        states = torch.cat(states, dim=0).cpu().numpy()
        actions = torch.cat(actions, dim=0).cpu().numpy()
        rewards = torch.cat(rewards, dim=0).cpu().numpy()
        cum_reward = rewards.sum() * env.scaler
        writer.add_scalar('cum_reward', cum_reward, info['episode'])

        fig = plt.figure(figsize=(16, 10))
        plt.title(f'Episode {info["episode"]} | Cumulative reward {cum_reward:.5e}')
        plt.axis('off')
        axis_t = np.arange(0, config.env.T_max)
        legends = ['T1', 'T2', 'T1I', 'T2I', 'V', 'E', 'a1', 'a2', 'reward']

        # states
        for i in range(6):
            ax = fig.add_subplot(3, 3, i+1)
            ax.plot(axis_t, states[:, i], label=legends[i])
            ax.legend()
            ax.grid()

        # actions
        for i in range(2):
            ax = fig.add_subplot(3, 3, i+7)
            ax.plot(axis_t, actions[:, i], label=legends[i+6], color='r')
            ax.legend()
            ax.grid()

        # rewards
        ax = fig.add_subplot(3, 3, 9)
        ax.plot(axis_t, rewards, label=legends[-1], color='g')
        ax.legend()
        ax.grid()

        evaldir = os.path.join(logdir, 'eval')
        os.makedirs(evaldir, exist_ok=True)
        fig.savefig(os.path.join(evaldir, f'result_{info["episode"]}.png'))
        plt.close()


