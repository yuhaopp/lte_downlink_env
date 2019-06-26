import datetime

from sac import *
from simulator import *


def plot_reward(name, reward_list, frame_idx):
    plt.figure(figsize=(10, 10))
    plt.plot(reward_list)
    plt.savefig('{}_{}.eps'.format(name, int(frame_idx / 10000)))
    plt.close()


def run(policy_net, policy_name, ue_arrival_rate=0.03, episode_tti=200.0):
    # environment
    env = Airview(ue_arrival_rate, episode_tti)

    # now we test the performance of trained policy
    max_frames = int(episode_tti * 1000)
    frame_idx = 0
    transmit_rate_list = []
    average_reward_list = []
    num_all_users_list = []
    num_selected_users_list = []

    while frame_idx < max_frames:
        state = env.reset()
        episode_reward = 0.0
        done = False
        record_flag = False
        record_number = 10

        while not done:
            action = policy_net.get_action(state)
            next_state, reward, done, all_buffer, num_all_users, num_selected_users = env.step(action)

            state = next_state
            episode_reward += reward
            frame_idx += 1
            if frame_idx % 1000 == 0:
                print(frame_idx)
            average_reward_list.append(episode_reward / frame_idx)
            transmit_rate_list.append(episode_reward / all_buffer)
            num_all_users_list.append(num_all_users)
            num_selected_users_list.append(num_selected_users)
            if frame_idx % 20000 == 0:
                time = str(datetime.datetime.now())
                plot_reward('test_{}_policy_reward'.format(policy_name), average_reward_list, frame_idx)
                log = open("test_{}_policy_result_{}.txt".format(policy_name, frame_idx), "w")
                log.write(str(average_reward_list))
                log.write('\n')
                log.write(str(transmit_rate_list))
                log.write('\n')
                log.write(str(num_all_users_list))
                log.write('\n')
                log.write(str(num_selected_users_list))
                log.close()
            if num_selected_users == 3:
                record_flag = True
            if record_flag and record_number > 0:
                record_number -= 1
                print("Frame ID: {}".format(str(frame_idx)))
                print("Current users: {}".format(str(num_selected_users)))

                action = action.reshape((RBG_NUM, MAX_MCS - MIN_MCS + 1))
                mcs_list = np.argmax(action, axis=-1) + 1
                print("MCS: {}".format(str(mcs_list)))

                print("Reward: {}".format(str(reward)))
