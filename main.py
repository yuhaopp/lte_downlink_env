import test_policy
import baseline_huawei_policy
import torch
import pandas as pd
import datetime
import model

ue_arrival_rate = 0.03
episode_tti = 200.0

# average_reward_list, qos_list = model.train(ue_arrival_rate, episode_tti)
#
# result = pd.DataFrame({'average_reward_list': average_reward_list, 'qos_list': qos_list})
# result.to_csv('train_result.csv', index=False)

rb_ddpg_policy_net = torch.load('RB_ddpg_policy_net.pth')
mcs_ddpg_policy_net = torch.load('MCS_ddpg_policy_net.pth')

average_reward_list, qos_list = model.test(rb_ddpg_policy_net, mcs_ddpg_policy_net, ue_arrival_rate, episode_tti)
test_result = pd.DataFrame({'average_reward_list': average_reward_list, 'qos_list': qos_list})
test_result.to_csv('test_result.csv', index=False)

#
# baseline_result = baseline_huawei_policy.run(ue_arrival_rate, episode_tti)
#
# time = str(datetime.datetime.now())
# result = pd.DataFrame({"train_result": train_result, "test_result": test_result, "baseline_result": baseline_result})
# result.to_csv('result_{}.csv'.format(time), index=False)
