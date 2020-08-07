# -*- coding:utf-8 -*-
import numpy as np
import collections

modelRange = collections.OrderedDict()

modelRange["alexnet"] = [2,2]
modelRange["resnet50"] = [6,10]
modelRange["googlenet"] = [4,5]
modelRange["inception-v3"] = [4,5]
modelRange["resnet18"] = [2,2]
modelRange["vgg16"] = [7,10]
modelRange["resnet34"] = [3, 3]
modelRange["vgg19"] = [10, 15]
modelRange["resnet101"] = [10,14]
modelRange["mobilenet"] = [3,4]

# 用于计算reward
coreJCT = collections.OrderedDict()
coreJCT["alexnet"] = [2.6]
coreJCT["resnet50"] = [3.7, 3.4, 3.3, 3.0, 2.7]
coreJCT["googlenet"] = [3.0, 2.6]
coreJCT["inception-v3"] = [4.1, 3.6]
coreJCT["resnet18"] = [2.8]
coreJCT["vgg16"] = [3.6, 3.2, 2.9, 2.6]
coreJCT["resnet34"] = [2.8]
coreJCT["vgg19"] = [4.7, 4.4, 4.1, 3.9, 3.7, 3.5]
coreJCT["resnet101"] = [5.1, 4.8, 4.4, 4.3, 3.9]
coreJCT["mobilenet"] = [3.6, 3.0]

relaCoreJCT = collections.OrderedDict()

for i in coreJCT:
    relaCoreJCT[i] = []
    for j in range(len(coreJCT[i])):
        relaCoreJCT[i].append(coreJCT[i][0] / coreJCT[i][j])

# 单机有5张卡，供160个核，这一批任务最多一共需要70个核，也就是说最少可以跑2轮；最少需要50个核，最多可以跑3轮

#设置最大的作业量为3轮，也就是每个做一个跑三次，
# action = {alexnet-1-2, alexnet-2-2, alexnext-3-2, .....}
model_number = 10
action_number = 0

for model in coreJCT:
    action_number += len(coreJCT[model])*3

one_batch_action_number = action_number / 3

action_set = [i for i in range(action_number)]

# 一个节点有五张卡，每个卡有32个核
card_number = 5
core_number = 32
# state_set = {card_0_0, card_0_1,...card_index_usedcore, ....}
state_set = [i for i in range(card_number * (core_number+1))]

def cardLeftResource(sid):
    return 32 - (sid - sid / 33 * 33)


# 一批作业一共有
def getUsedCoreAndJCTByActionId(actionId):
    curIdx = 0
    for i in coreJCT:
        for m in range(len(coreJCT[i])):
            for j in range(3):
                if curIdx == actionId:
                    return modelRange[i][0] + m, relaCoreJCT[i][m]
                curIdx += 1
    print("don't find model \n ")
    return -1


# 最后一个卡资源为0的时候
def hasResource(sid):
    return sid <= 159

def getReward(jct):
    return 1 + jct


def Qlearning():
    # hyperparameter
    gamma = 0.8
    epsilon = 0.4

    # Q(state, action) = Reward(state, action) + Gamma*Max(Q(state+1, all actions))
    Q = np.zeros([len(state_set), len(action_set)])
    print len(action_set), len(state_set)
    for episode in range(201):
        # 从第一张卡开始选择
        state = 0
        # 使用过的action需要删除
        possible_actions = action_set
        # 当没有可用core时为终止条件
        while (hasResource(state)):

            # Step next state, here we use epsilon-greedy algorithm.
            if np.random.random() < epsilon:
                # choose random action
                action = possible_actions[np.random.randint(0, len(possible_actions))]
            else:
                # greedy
                action = np.argmax(Q[state])

            # 删除使用过的动作
            possible_actions = action_set[:action] + action_set[action+1:]

            # Update Q value
            usedCore, jct = getUsedCoreAndJCTByActionId(action)
            if cardLeftResource(state):
                reward = getReward(jct)
            else:
                #reward = -1
                reward = getReward(jct)
            Q[state, action] = reward + gamma * Q[state].max()

            # Go to the next state
            state += usedCore

        # Display training progress
        if episode % 10 == 0:
            print("------------------------------------------------")
            print("Training episode: %d" % episode)
            print(Q)

    # save Q matrix
    qf = open("Q-matrix.csv", "w+")
    [rows, cols] = Q.shape
    for i in range(rows):
        for j in range(cols):
            qf.write(str(Q[i][j]))
            qf.write("\t")
        qf.write("\n")


if __name__ == "__main__":
    Qlearning()