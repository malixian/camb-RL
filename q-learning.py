# -*- coding:utf-8 -*-
import numpy as np


#State（Memory、Utilization、Reference Time）
#Action （mp、dp、batch_size）


# 为了简化运算，定义以下规则，此规则映射了每个状态可采取的动作：
# Memory（M）-> mp
# Utilization（U）-> dp
# Reference_time（T）-> batch_size
# 状态转移流程为 M->U->R

# reward，col represent action，row represent state

"""
Action:
mps = [1, 2, 4, 8, 16, 32]

batchs = [1, 2, 4]

dataparallels = [1, 2, 4, 8, 16, 32]

================
State:
可用的内存 Memory: [1，2，4，8，16， 32]（与数据并行度一一对应，num*单个模型需要的内存）
可用的core：[1, 2, ...., 32]
推理的时间：[1, 2, 3] , 1代表good， 2代表normal，3代表bad
"""

# reward matrix
# 内存使用量越多分数越低、core使用越多分数越低、推理时间越短分数越高
# 当前reward值与前一个状态相关
# (M, mp) = 1/mem; (U, dp) = 1/(dp*mp); (T, batch_size) = 1/reference_time
# Actions：
mps = [1, 2, 4, 8, 16]
batchs = [1, 2, 4]
dps = [1, 2, 4, 8, 16]

"""
action set 定义为：
NO	mp	dp	score(reward, throughput/(mp*dp*time))
0	1	1	1.256 
1	1	2	2.091 
2	1	4	2.980 
3	1	8	3.539 
4	1	16	5.318 
5	2	1	1.695 
6	2	2	2.890 
7	2	4	3.894 
8	2	8	2.737 
9	2	16	1.154 
10	4	1	1.861 
11	4	2	2.212 
12	4	4	2.395 
13	4	8	1.221 
14	8	1	1.526 
15	8	2	1.632 
16	8	4	1.438 
17	16	1	0.877 
18	16	2	0.776 
"""


action_set = [i for i in range(19)]

"""


"""

action_reward_map = {
    0   :   4.954809418,
    1   :   8.51707657,
    2   :   12.32754352,
    3   :   12.85317121,
    4   :   9.161624003,
    5   :   6.36743321,
    6   :   10.53608601,
    7   :   12.6275721,
    8   :   7.348988945,
    9   :   2.797860738,
    10  :   7.060278158,
    11  :   7.522317377,
    12  :   8.550589954,
    13  :   4.776430015,
    14  :   5.443691438,
    15  :   5.317585384,
    16  :   4.502125077,
    17  :   3.393294033,
    18  :   2.514767144

}

action_mpdp_map = [1,	1, #action_set 0
                  1,	2, #action_set 1 ...
                  1,	4, #action_set 2
                  1,	8, #action_set 3
                  1,	16, #action_set 4
                  2,	1, #action_set 5
                  2,	2, #action_set 6
                  2,	4, #action_set 7
                  2,	8, #action_set 8
                  2,	16, #action_set 9
                  4,	1, #action_set 10
                  4,	2, #action_set 11
                  4,	4, #action_set 12
                  4,	8, #action_set 13
                  8,	1, #action_set 14
                  8,	2, #action_set 15
                  8,    4, #action_set 16
                  16,	1, #action_set 17
                  16,	2  #action_set 18
]

# States:
mem_s = [1, 2, 4, 8, 16, 32]
core_s = [i for i in range(1, 33)]

# reward 矩阵初始化
row = len(core_s)
col = len(action_set)
reward = np.zeros([row, col])
#init reward matrix
for cur_state in core_s:
    for action in action_set:
        if cur_state >= action_mpdp_map[2*action] * action_mpdp_map[2*action+1]:
            reward[cur_state-1][action] = action_reward_map[action]
        else:
            reward[cur_state-1][action] = 0


"""
保存reward矩阵
wf = open("reward_matrix_1.csv", 'w+')
for cur_state in core_s:
    for action in action_set:
        wf.write(str(reward[cur_state-1][action]))
        wf.write("\t")
    wf.write("\n")
wf.close()
"""
# hyperparameter
gamma = 0.8
epsilon = 0.4

#Q(state, action) = Reward(state, action) + Gamma*Max(Q(state+1, all actions))
Q = np.zeros([row, col])
MAX_UNUSED_CORE = 32

for episode in range(201):
    # 随机选择一个状态
    state = np.random.randint(1, 32)
    unused_core = MAX_UNUSED_CORE - state
    # 当没有可用core时为终止条件
    while(unused_core > 0):
        possible_actions = []
        possible_q = []
        for action in action_set:
            if(reward[state][action] > 0):
                possible_actions.append(action)
                possible_q.append(Q[state, action])

        # Step next state, here we use epsilon-greedy algorithm.
        action = -1
        if np.random.random() < epsilon:
            # choose random action
            action = possible_actions[np.random.randint(0, len(possible_actions))]
        else:
            # greedy
            action = possible_actions[np.argmax(possible_q)]

        # Update Q value
        Q[state, action] = reward[state, action] + gamma * Q[action].max()

        # Go to the next state

        state = action_mpdp_map[action*2] * action_mpdp_map[action*2+1]
        unused_core = unused_core - state

    # Display training progress
    if episode % 10 == 0:
        print("------------------------------------------------")
        print("Training episode: %d" % episode)
        print(Q)

# save Q matrix
qf = open("Q-matrix-resnet50 .csv", "w+")
[rows, cols] = Q.shape
for i in range(rows):
    for j in range(cols):
        qf.write(str(Q[i][j]))
        qf.write("\t")
    qf.write("\n")





