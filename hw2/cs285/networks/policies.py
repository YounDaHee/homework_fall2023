import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        obs_ = ptu.from_numpy(obs)
        dist = self.forward(obs_)
        if self.discrete: 
            action = dist.sample()
        else :
            action = dist.rsample()

        return ptu.to_numpy(action)
    
    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            # 타 코드 기반으로 수정
            # torch.distribution 활용하여 랜덤하게 "탐험"할 수 있도록 유도
            logits = self.logits_net(obs)
            dist = distributions.Categorical(logits = logits)
            
            '''
            logits = self.logits_net(obs)
            logits = torch.softmax(logits, dim = -1)

            # softmax로 추출한 확률값으로부터 랜덤하게 추출
            dist = torch.distributions.Categorical(probs=logits)
            acs = dist.sample()

            # 가장 큰 값 추출
            #acs = torch.argmax(logits)
            
            # 이론상 가장 큰 값을 추출해도 정상적으로 학습은 이루어져야됨(안좋은 reward에 대해 패널티를 주기 때문)
            # 하지만, baseline이 없는 버전에서 nomalize를 하지 않으면 나쁜 결과에도 reward 부여
            # 결과적으로 좋지 않은 선택이지만 선택 확률이 증가하는 모순 발생
            # 하지만 랜덤으로 선택 시 우연히 좋은 선택 수행하면, 해당 행동 선택 확률 증가.
            # 좋은 선택을 더 많이 선택하도록 유도.
            # baseline이 없는 버전에서도 학습에 따라 결과가 좋아지게 됨.
            '''
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            mean = self.mean_net(obs)
            dist = distributions.Normal(loc = mean, scale = torch.exp(self.logstd))
            
        return dist

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        N : int
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.

        dist = self.forward(obs)
        
        if self.discrete :
            '''
            logits = self.logits_net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions.squeeze())
            loss = -torch.mean(log_probs * advantages)
            '''
            '''
            logits = self.logits_net(obs)
            log_probs = torch.nn.functional.cross_entropy(logits, actions.long().squeeze(), reduction='none')
            loss = torch.mean(log_probs * advantages)
            '''
            
            loss = -torch.mean(dist.log_prob(actions)*advantages)
        else :
            '''
            mean = self.mean_net(obs)  # 행동 평균(정확히는 기댓값)
            std = torch.exp(self.logstd)  # 분산

            # mean과 std를 이용하여 가우시안 분포 획득
            dist = torch.distributions.Normal(mean, std)
            # 가우시안 분포로 얻은 행동에 대해 행동의 log 분포 획득
            log_probs = dist.log_prob(actions).sum(axis=-1) 

            loss = -torch.mean(log_probs * advantages)

            #logits = self.mean_net(obs)
            #loss = -torch.mean(torch.log(logits)*advantages.unsqueeze(1))
            '''

            loss = -torch.mean(dist.log_prob(actions).sum(dim=-1)*advantages)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
