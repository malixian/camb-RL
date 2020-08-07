# -*- coding:utf-8 -*-
import numpy as np
import collections, copy

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
        relaCoreJCT[i].append(coreJCT[i][len(coreJCT[i])-1] / coreJCT[i][j] )

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

def getModelByActionId(actionId):
    curIdx = 0
    for i in coreJCT:
        for m in range(len(coreJCT[i])):
            for j in range(3):
                if curIdx == actionId:
                    return i
                curIdx += 1
    print("don't find model \n ")
    return ""

# 最后一个卡资源为0的时候
def hasResource(sid):
    return sid <= card_number * (core_number+1) - 1

def getReward(jct):
    return 2 + jct


def Qlearning():
    # hyperparameter
    gamma = 0.8
    epsilon = 0.4

    # Q(state, action) = Reward(state, action) + Gamma*Max(Q(state+1, all actions))
    Q = np.zeros([len(state_set), len(action_set)])
    max_reward_trace = []
    max_reward = 0
    for episode in range(5000):
        acc_reward = 0
        # 从第一张卡开始选择
        state = 0
        # 使用过的action需要删除
        possible_actions = copy.deepcopy(action_set)
        possible_q = copy.deepcopy(Q)
        # 当没有可用core时为终止条件
        trace = []
        model_count = {}
        for i in modelRange:
            model_count[i] = 3
        while hasResource(state)  and len(possible_actions) > 0:
            # Step next state, here we use epsilon-greedy algorithm.
            if np.random.random() < epsilon:
                # choose random action
                action = possible_actions[np.random.randint(0, len(possible_actions))]
            else:
                # greedy
                max_q_action = possible_actions[np.random.randint(0, len(possible_actions))]
                max_q = 0.0
                for to_action in range(len(possible_q[state])):
                    if model_count[getModelByActionId(to_action)] == 0:
                        continue
                    if possible_q[state][to_action] >= max_q:
                        max_q = possible_q[state][max_q_action]
                        max_q_action = to_action
                action = max_q_action
            trace.append(action)

            # 统计作业个数
            model = getModelByActionId(action)
            model_count[model] = model_count[model] - 1
            if model_count[model] == 0:
                # 删除所有action对应的model
                for i in possible_actions:
                    if getModelByActionId(i) == model:
                        possible_actions.remove(i)

            # Update Q value
            usedCore, jct = getUsedCoreAndJCTByActionId(action)
            if cardLeftResource(state):
                reward = getReward(jct)
            else:
                #reward = -1
                reward = getReward(jct)
            acc_reward += reward
            Q[state, action] = reward + gamma * Q[state].max()

            # Go to the next state
            state += usedCore
        if max_reward < acc_reward:
            max_reward_trace = trace

        # Display training progress
        if episode >=100 and episode % 100 == 0:
            print("------------------------------------------------")
            #print("Training episode: %d" % episode)
            #print(Q)
            print("iter %d reward is %d \n" % (episode, acc_reward))
            #print trace
            #for i in trace:
            #    print getModelByActionId(i)
            #    print getUsedCoreAndJCTByActionId(i)
    # save Q matrix
    print "max trace is", max_reward_trace
    all_used_core = 0
    for i in trace:
        print getModelByActionId(i)
        used_core, jct = getUsedCoreAndJCTByActionId(i)
        all_used_core += used_core
    print "max reward is ", max_reward
    print "all used core is ", all_used_core
    qf = open("Q-matrix-" + str(card_number) + "card.csv", "w+")
    [rows, cols] = Q.shape


    for i in range(rows):
        for j in range(cols):
            qf.write(str(Q[i][j]))
            qf.write("\t")
        qf.write("\n")


if __name__ == "__main__":
    Qlearning()
