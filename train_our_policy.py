from sac import *
from simulator import *


def plot_reward(name, reward_list, frame_idx):
    plt.figure(figsize=(10, 10))
    plt.plot(reward_list)
    plt.savefig('{}_{}.eps'.format(name, int(frame_idx / 10000)))
    plt.close()


# train our networks
def run(ue_arrival_rate=0.04, episode_tti=200.0):
    # environment
    env = Airview(ue_arrival_rate, episode_tti)
    state_dim = env.state_dim
    action_dim = env.action_dim
    hidden_dim = 512
    batch_size = 512

    # sac algorithm
    sac = SoftActorCritic(state_dim, action_dim, hidden_dim)

    max_frames = int(episode_tti * 1000)
    frame_idx = 0
    average_reward_list = []
    transmit_rate_list = []
    num_all_users_list = []
    num_selected_users_list = []

    while frame_idx < max_frames:
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = sac.policy_net.get_action(state)
            next_state, reward, done, all_buffer, num_all_users, num_selected_users = env.step(action)

            sac.replay_buffer.push(state, action, reward, next_state, done)
            if len(sac.replay_buffer) > batch_size:
                sac.soft_q_update(batch_size)

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
                print('current users: {}'.format(num_all_users_list[frame_idx - 1]))
                plot_reward('our_policy_reward', average_reward_list, frame_idx)
                log = open("train_policy_result_{}.txt".format(frame_idx), "w")
                log.write(str(average_reward_list))
                log.write('\n')
                log.write(str(transmit_rate_list))
                log.write('\n')
                log.write(str(num_all_users_list))
                log.write('\n')
                log.write(str(num_selected_users_list))
                log.close()

    torch.save(sac.policy_net, 'sac_policy_net.pth')
