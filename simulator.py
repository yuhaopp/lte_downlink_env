# -*- coding: utf-8 -*-

import numpy as np
import copy
from sklearn.preprocessing import minmax_scale

"""
Configuration:
"""
BANDWIDTH = 1e7
RBG_NUM = 17
TTI = 0.001
UE_ARRIVAL_RATE = 0.03
PACKET_SIZE = (int(1e3), int(1e6))
CQI_REPORT_INTERVAL = 0.02
THROUGHPUT_BASELINE = 10000.0
MIN_MCS = 1
MAX_MCS = 29
EPISODE_TTI = 10.0
FREQUENCY = 2e9
LIGHT_SPEED = 3e8


class User:
    var = {
        'num': ['buffer', 'rsrp', 'avg_snr', 'avg_thp'],
        'vec': ['cqi', 'se', 'prior', 'sched_rbg']
    }
    attr_range = {
        'buffer': (int(1e3), int(1e6)),
        'rsrp': (-120, -90),
        'avg_snr': (1, 31),
        'avg_thp': (0, BANDWIDTH * 0.9 * TTI * np.log2(1 + 29 ** 2)),
        'cqi': (1, 29),
        'mcs': (1, 29),
        'se': tuple(map(lambda x: np.log2(1 + x ** 2), (1, 29))),
        'prior': (0, np.log2(1 + 29 ** 2)),
        'sched_rbg': (0, 1)
    }

    def __init__(self, user_id, arr_time, buffer, rsrp, is_virtual=False):
        self.ID = user_id
        self.arr_time = arr_time
        self.buffer = buffer
        self.rsrp = rsrp
        self.avg_snr = self.rsrp + 121
        self.avg_thp = 0
        self.cqi = np.full(RBG_NUM, np.nan)
        self.mcs = np.full(RBG_NUM, np.nan)
        self.se = np.full(RBG_NUM, np.nan)
        self.sched_rbg = np.zeros(RBG_NUM)
        self.is_virtual = is_virtual
        self.sum_reward = 0
        self.olla_enable = False
        self.olla_value = 0.0
        self.olla_step = 0.01
        self.speed = 3
        self.coherence_time = LIGHT_SPEED / (2.0 * self.speed * FREQUENCY)
        self.cqi_update_time = None

    def reset_rbg(self):
        self.sched_rbg.fill(0)

    def __getitem__(self, x):
        return getattr(self, x)

    def __setitem__(self, key, value):
        self.__dict__[key] = value


"""
Here we set the state of environment from the perspective of RBGs
The dimension is 17*(attr)
attr is the features of user who is assigned to this RBG, which includes:
buffer: the package size of the user
rsrp: the user's rsrp
avg_snr: the user's avg_snr
avg_thp: the user' avg_thp
cqi: the user's cqi in this RBG
se: the user's se in this RBG
prior: the user's prior in this RBG
"""

"""
user_list: all active users
select_user_list: all scheduled users
count_user: number of all arrived users
rb_state_dim: dimension of RB policy state
rb_action_dim: dimension of RB policy action
mcs_state_dim: dimension of MCS policy state
mcs_action_dim: dimension of MCS policy action
"""


class Airview():
    def __init__(self, ue_arrival_rate=UE_ARRIVAL_RATE, episode_tti=EPISODE_TTI):
        self.ue_arrival_rate = ue_arrival_rate
        self.cqi_report_interval = CQI_REPORT_INTERVAL

        self.episode_tti = episode_tti
        self.packet_list = np.random.uniform(1e3, 1e6, int(self.episode_tti * 1000))
        self.rsrp_list = np.random.uniform(-120, -90, int(self.episode_tti * 1000))

        self.sim_time = 0.0
        self.all_buffer = 0

        self.user_list = []
        self.select_user_list = []
        self.count_user = 0

        self.rb_state_dim = RBG_NUM * RBG_NUM * 4
        self.rb_action_dim = RBG_NUM * RBG_NUM
        self.rb_state_list = []

        self.mcs_state_dim = 1
        self.mcs_action_dim = MAX_MCS - MIN_MCS + 1
        self.mcs_state_list = []

        self.system_log = []

    def reset(self):
        self.__init__(self.ue_arrival_rate, self.episode_tti)
        self.fill_in_vir_users()
        self.add_new_user()
        self.update_cqi()
        return self.get_rb_state()

    def fill_in_vir_users(self):
        fill_count = max(0, RBG_NUM - len(self.user_list))
        for i in range(fill_count):
            self.user_list.append(User(-1.0, self.sim_time, 1.0, -120.0, True))
            self.all_buffer += 1

    def add_new_user(self):
        if np.random.uniform(0., 1.) < self.ue_arrival_rate:
            self.count_user += 1
            user = User(self.count_user, self.sim_time, self.packet_list[self.count_user],
                        self.rsrp_list[self.count_user])
            self.all_buffer += user.buffer
            if not self.user_list[len(self.user_list) - 1].is_virtual:
                self.user_list.append(user)
                return

            for i in range(len(self.user_list)):
                if self.user_list[i].is_virtual:
                    self.user_list[i] = user
                    break

    def update_cqi(self):
        for i in range(len(self.user_list)):
            user = self.user_list[i]
            live_time = self.sim_time - user.arr_time
            if live_time % self.cqi_report_interval == 0.0:
                '''random implies error in channel measurement'''
                user.cqi = user.avg_snr + np.random.randint(-2, 2, size=RBG_NUM)
                user.cqi = np.clip(user.cqi, *user.attr_range['cqi'])
                user.cqi_update_time = self.sim_time
                # if user is virtual, then cqi is set to 0.
                if user.is_virtual:
                    user.cqi = np.zeros(RBG_NUM)

            user.mcs = copy.deepcopy(user.cqi)
            user.se = np.log2(1 + user.mcs ** 2.0)
            self.user_list[i] = user

    '''get state list for RB policy'''

    def get_rb_state(self):
        self.rb_state_list = []
        for i in range(RBG_NUM):
            rb_i_state = []
            for j in range(RBG_NUM):
                user = self.user_list[j]
                scaled_avg_snr = (user.avg_snr - 1.0) / (31.0 - 1.0)
                scaled_cqi = (user.cqi[i] - 1.0) / (29.0 - 1.0)
                scaled_buffer = (user.buffer - 1e3) / (1e6 - 1e3)
                scaled_avg_thp = (user.avg_thp - 1e4) / 1e4

                rb_i_state.extend([scaled_avg_snr, scaled_cqi, scaled_buffer, scaled_avg_thp])
            self.rb_state_list.extend(rb_i_state)
        return self.rb_state_list

    '''receive action list for RB policy and take action'''

    def take_rb_action(self, rb_action_list):
        rb_action_list = np.reshape(rb_action_list, (RBG_NUM, RBG_NUM))
        rb_action_list = np.argmax(rb_action_list, axis=1)
        sched_user_set = set()
        for i in range(len(rb_action_list)):
            index = rb_action_list[i]
            self.user_list[index].sched_rbg[i] = 1
            sched_user_set.add(self.user_list[index])
        self.select_user_list = list(sched_user_set)

    '''get state list for MCS policy'''

    def get_mcs_state(self):
        self.mcs_state_list = []
        for i in range(len(self.select_user_list)):
            select_user = self.select_user_list[i]
            state = np.array([select_user.avg_snr])
            self.mcs_state_list.append(state)
        return self.mcs_state_list

    '''receive action list for MCS policy and take action'''

    def take_mcs_action(self, mcs_action_list):
        system_reward = 0
        mcs_reward_list = []
        for i in range(len(self.select_user_list)):
            user = self.select_user_list[i]
            mcs_action_list[i] += int(user.olla_value) if user.olla_enable else 0
            if user.cqi_update_time is not None:
                time_decorrelation = min((1, int((self.sim_time - user.cqi_update_time) / user.coherence_time))) * 2
            else:
                time_decorrelation = 2

            rand_factor = 0 if time_decorrelation == 0 else np.random.randint(-time_decorrelation,
                                                                              time_decorrelation)

            is_succ = 1 if (user.avg_snr + rand_factor - mcs_action_list[i]) > 0 else 0
            user.olla_value += user.olla_step if (is_succ == 1) else -9.0 * user.olla_step
            rbg_se = np.log2(1 + mcs_action_list[i] ** 2)
            rbg_tbs = int(BANDWIDTH * 0.9 * user.sched_rbg.sum() / RBG_NUM * rbg_se * TTI)

            if rbg_tbs > user.buffer:
                rbg_tbs = user.buffer
                user.buffer = 0
            else:
                user.buffer -= rbg_tbs

            reward = rbg_tbs * is_succ
            user.sum_reward += reward
            user.avg_thp = user.sum_reward / ((self.sim_time - user.arr_time) * 1000 + 1)
            self.select_user_list[i] = user

            if user.buffer == 0 and user.sum_reward > 1:
                self.system_log.append([user.arr_time, self.sim_time, user.sum_reward, user.avg_thp])

            system_reward += reward
            mcs_reward_list.append(reward / user.sched_rbg.sum())
        rb_reward = self.get_rb_reward()
        return system_reward, mcs_reward_list, rb_reward

    '''get rewards for RB policy and MCS policy'''

    def get_rb_reward(self):
        reward = 0.0
        num_active_user = self.get_num_active_users()

        if num_active_user != 0:
            for i in range(num_active_user):
                avg_thp = self.user_list[i].avg_thp
                single_reward = 1 if avg_thp > 10000 else 0
                reward += single_reward
            reward = reward / float(num_active_user)
        return reward

    def del_empty_user(self):
        self.user_list = list(filter(lambda x: x.buffer > 0, self.user_list))

    '''get the number of active users and scheduled users'''

    def get_num_active_users(self):
        num = 0
        for user in self.user_list:
            if user.ID != -1:
                num += 1
        return num

    '''after take rb action, the algorithm run step'''

    def step(self, mcs_action_list):
        system_reward, mcs_reward_list, rb_reward = self.take_mcs_action(mcs_action_list)

        updated_mcs_state_list = []
        for i in range(len(self.select_user_list)):
            updated_mcs_state_list.append([np.random.uniform(1, 31)])

        # check current number of active/selected users
        num_active_users = self.get_num_active_users()
        num_selected_users = len(self.select_user_list)

        self.sim_time += TTI

        # del user with empty buffer
        self.del_empty_user()

        # new user comes with probability
        self.add_new_user()

        # create virtual users to fill in user_list, this will be executed only when len(user_list)<RBG_NUM
        self.fill_in_vir_users()

        # update cqi
        self.update_cqi()

        done = int(self.sim_time) == int(self.episode_tti)

        # get new state
        next_rb_state_list = self.get_rb_state()
        next_mcs_state_list = self.get_mcs_state()

        return next_rb_state_list, rb_reward, updated_mcs_state_list, next_mcs_state_list, mcs_reward_list, system_reward, done, self.all_buffer, num_active_users, num_selected_users

    def get_system_log(self):
        return self.system_log

    def get_action(self):
        # reward by Huawei Policy, compare with the policy network we trained
        # mcs_list = [np.floor(np.sum(ue.mcs * ue.sched_rbg) / ue.sched_rbg.sum()) for ue in
        #             self.select_user_list]
        # for i in range(len(mcs_list)):
        #     mcs_list[i] -= 3
        # action = np.zeros((RBG_NUM, MAX_MCS - MIN_MCS + 1))
        # for i in range(len(mcs_list)):
        #     action[i][int(mcs_list[i] - 1)] = 1
        # action = action.reshape(RBG_NUM * (MAX_MCS - MIN_MCS + 1))

        snr_list = np.array([ue.avg_snr for ue in self.select_user_list])
        snr_list = np.clip(snr_list - 3, 1, 29)
        return snr_list
