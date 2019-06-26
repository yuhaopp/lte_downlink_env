import train_sac_policy
import test_policy
import baseline_huawei_policy
from ddpg import *
import torch

ue_arrival_rate = 0.03
episode_tti = 200.0

ddpg = DDPG(ue_arrival_rate, episode_tti)
ddpg.run()
ddpg_policy_net = torch.load('ddpg_policy_net.pth')
test_policy.run(ddpg_policy_net, 'ddpg_policy', ue_arrival_rate, episode_tti)

baseline_huawei_policy.run(ue_arrival_rate, episode_tti)

# train_sac_policy.run(ue_arrival_rate, episode_tti)
# sac_policy_net = torch.load('sac_policy_net.pth', map_location='cpu')
# test_policy.run(sac_policy_net, 'sac_policy', ue_arrival_rate, episode_tti)
