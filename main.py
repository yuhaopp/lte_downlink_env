import test_policy
import baseline_huawei_policy
from ddpg import *
import torch
import pandas as pd
import datetime

ue_arrival_rate = 0.03
episode_tti = 200.0

ddpg = DDPG(ue_arrival_rate, episode_tti)
train_result = ddpg.run()

ddpg_policy_net = torch.load('ddpg_policy_net.pth')
test_result = test_policy.run(ddpg_policy_net, 'ddpg_policy', ue_arrival_rate, episode_tti)

baseline_result = baseline_huawei_policy.run(ue_arrival_rate, episode_tti)

time = str(datetime.datetime.now())
result = pd.DataFrame({"train_result": train_result, "test_result": test_result, "baseline_result": baseline_result})
result.to_csv('result_{}.csv'.format(time), index=False)
