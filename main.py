import test_policy
import baseline_huawei_policy
from ddpg import *
import torch

ue_arrival_rate = 0.03
episode_tti = 200.0
train_reward_list = []
test_reward_list = []
baseline_reward_list = []

ddpg = DDPG(ue_arrival_rate, episode_tti)
ddpg.run()

ddpg_policy_net = torch.load('ddpg_policy_net.pth')
test_policy.run(ddpg_policy_net, 'ddpg_policy', ue_arrival_rate, episode_tti)
baseline_huawei_policy.run(ue_arrival_rate, episode_tti)
