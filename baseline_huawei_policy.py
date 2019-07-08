from sac import *
from simulator import *
import datetime


def plot_reward(name, reward_list, frame_idx):
    plt.figure(figsize=(10, 10))
    plt.plot(reward_list)
    plt.savefig('{}_{}.eps'.format(name, int(frame_idx / 10000)))
    plt.close()


def random_policy():
    return np.random.randn(493)


def run(ue_arrival_rate=0.04, episode_tti=200.0):
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
        env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = env.get_action()
            next_state, reward, done, all_buffer, num_all_users, num_selected_users = env.step(action)

            episode_reward += reward
            frame_idx += 1
            if frame_idx % 1000 == 0:
                print(frame_idx)
            average_reward_list.append(episode_reward / frame_idx)
            transmit_rate_list.append(episode_reward / all_buffer)
            num_all_users_list.append(num_all_users)
            num_selected_users_list.append(num_selected_users)
            if frame_idx % 200000 == 0:
                time = str(datetime.datetime.now())
                plot_reward('baseline_policy_result_{}'.format(time), average_reward_list, frame_idx)
                log = open("baseline_policy_result_{}_{}.txt".format(frame_idx, time), "w")
                log.write(str(average_reward_list))
                log.write('\n')
                log.write(str(transmit_rate_list))
                log.write('\n')
                log.write(str(num_all_users_list))
                log.write('\n')
                log.write(str(num_selected_users_list))
                log.close()
    return average_reward_list
