import torch
import pandas as pd
import model

ue_arrival_rate = 0.03
episode_tti = 200.0
episode_number = 1

average_reward_list, qos_list, system_log, time = model.train(ue_arrival_rate, episode_tti, episode_number)

rb_ddpg_policy_net = torch.load('RB_ddpg_policy_net_{}.pth'.format(time))
mcs_ddpg_policy_net = torch.load('MCS_ddpg_policy_net_{}.pth'.format(time))

test_average_reward_list, test_qos_list, test_system_log = model.test(rb_ddpg_policy_net, mcs_ddpg_policy_net,
                                                                      ue_arrival_rate,
                                                                      episode_tti)

result = pd.DataFrame({'average_reward_list': average_reward_list, 'qos_list': qos_list})
result.to_csv('result_{}.csv'.format(time), index=False)
train_system_log = pd.DataFrame(system_log)
train_system_log.to_csv('train_system_log_{}.csv'.format(time), index=False)

test_result = pd.DataFrame({'test_average_reward_list': test_average_reward_list, 'test_qos_list': test_qos_list})
test_result.to_csv('test_result_{}.csv'.format(time), index=False)
test_system_log = pd.DataFrame(test_system_log)
test_system_log.to_csv('test_system_log_{}.csv'.format(time), index=False)
