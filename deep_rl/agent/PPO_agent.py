#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import random


class PPOAgent(BaseAgent):
    # PPO agent config
    def generate_config(utils):
        config = utils.config.Config()
        config.num_workers = 20
        config.state_dim = 33
        config.action_dim = 4
        config.network_fn = lambda: GaussianActorCriticNet(
            config.state_dim, config.action_dim,
            actor_body=network_bodies.FCBody(config.state_dim, hidden_units=(64, 64), gate=torch.tanh),
            critic_body=network_bodies.FCBody(config.state_dim, hidden_units=(64, 64), gate=torch.tanh))
        config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
        config.discount = 0.99
        config.use_gae = True
        config.gae_tau = 0.95
        config.gradient_clip = 0.5
        config.rollout_length = 256 #128
        config.optimization_epochs = 10
        config.mini_batch_size = 64*20 #32*20
        config.ppo_ratio_clip = 0.2
        config.max_steps = 1e6
        config.state_normalizer = utils.normalizer.MeanStdNormalizer()
        config.seed = 0
        return config

    def __init__(self, config, env):
        BaseAgent.__init__(self, config)
        self.config = config
        self.env = env
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.env.reset()
        self.states = config.state_normalizer(self.states)
        self.scores = np.zeros(20)
        self.total_episodes = 0
        self.seed = random.seed(config.seed)
        self.mean_score_history = []

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            next_states, rewards, terminals, info = self.env.step(to_np(prediction['a']))
            self.scores += rewards
            if np.any(terminals):
                self.total_episodes += 1
                print('# episode: {} --- total score (averaged over agents) this episode: {}'.format(self.total_episodes, np.mean(self.scores)))
                self.mean_score_history.append(np.mean(self.scores))
                self.env.reset()
                self.scores = np.zeros(20)


            #self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         's': tensor(states)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        states, actions, log_probs_old, returns, advantages = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = self.network(sampled_states, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()

                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()
