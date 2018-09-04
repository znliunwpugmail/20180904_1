from Imitater.DQN_net import dqn
import numpy as np
import random
import copy
import time

np.set_printoptions(suppress=True)
EPISDOE = 400000
STEP = 10000
LAMBDA = 0.1

class Devision():
    def __init__(self):
        self.capacility_states = [500,500,10]
        self.states = []#[U,C,e]存储
        self.user_states = []#U存储
        self.es = []#e存储
        self.inouts = []#进来、出去的标志，in-1，out-0
        self.id = []#用户id，在数据中，用户id一般出现两次，进来出现一次，出去出现一次。
        self.actions = []#用户选择什么action，长度是文件中数据的数量
        self.id_actions_dict = {}#键值对，用户id-action，每一个用户id选择了什么行为
        self.reject = None
        self.all_capacilitys = None


    def get_states(self,filename="user_imitater_u_e_final.txt"):#读取文件，生成U,e，并读取每一条数据的inout，和用户id.
        with open(filename) as f:
            line = f.readline()
            while line:
                content_str = line.split('\t')
                if(len(content_str)<6):
                    continue
                user_state = [float(content_str[3]),float(content_str[4]),float(content_str[5])]
                inout = int(content_str[2])
                e = [int(content_str[6])-int(content_str[7])]
                self.user_states.append(user_state)
                self.es.append(e)
                self.inouts.extend([inout])
                self.actions = []
                self.id.append(int(content_str[1]))
                line = f.readline()
        self.actions = np.ones(shape=(len(self.user_states)),dtype=np.int)*(-1)#cation初始化为-1
        self.reject = np.ones(shape=(len(self.user_states)),dtype=np.int)*(-1)
        self.all_capacilitys = np.ones(shape=(len(self.user_states),3),dtype=np.int)*(-2)
        for  i in range(len(self.user_states)):
            self.user_states[i][0:3] = [1,1,1]

    def relu(self,x):
        """Compute softmax values for each sets of scores in x."""
        x = np.array(x)
        x[x<0] = 0
        return x
    def train(self):
        all_steps = 0
        state_index = 0
        action_dim = 3
        state_dim = action_dim * 2 + 1
        agent = dqn.DQN(action_dim, state_dim)
        in_num = 0
        out_num = 0

        while 1 == 1:
            total_reward = 0

            user_state = self.user_states[state_index]
            e_state = self.es[state_index]
            inout =self.inouts[state_index]#以上获取U,e,inout
            # print(self.inouts)
            if 0 == inout:#如果inout状态为离开，相应action+1
                # print('inout',inout)
                # print(state_index)
                user_id = self.id[state_index]
                if user_id in self.id_actions_dict.keys():
                    action_out = self.id_actions_dict[user_id]#根据user_id获取相应的action
                    # print(action_out)
                    self.capacility_states[action_out[0]] += 1
                state_index+=1
                if state_index >= len(self.user_states):
                    break;
                state_index=state_index%len(self.user_states)
                out_num+=1
                continue
            state = copy.deepcopy(user_state)
            state.extend(self.capacility_states)
            state.extend(e_state)
            in_num+=1
            # print(state_index)
            for episode in range(len(self.user_states)):
                print(len(self.user_states))
                starttime = time.time()
                # print('1',state_index)
                all_steps+=1
                inout = self.inouts[state_index]
                if 0 == inout:#如果inout状态为离开，相应action+1
                    user_id = self.id[state_index]
                    if user_id in self.id_actions_dict.keys():
                        action_out = self.id_actions_dict[user_id]#找出对应的user_id的action
                        self.capacility_states[action_out[0]] += 1
                    state_index += 1
                    if state_index>=len(self.user_states):
                        break;
                    state_index = state_index % len(self.user_states)
                    out_num += 1
                    continue
                # state[3:6] = np.log10(state[3:6])
                action = agent.get_action([state])#将state送入dqn，执行dqn网络，生成action
                u = self.user_states[state_index][action[0]]
                p = random.random()
                # if p>u:
                #     self.reject[state_index] = copy.deepcopy(action[0])
                #     action[0] = 2
                # else:
                #     self.reject[state_index] = -1
                self.actions[state_index] = action[0]
                user_id = self.id[state_index]
                # print('action', action)
                self.capacility_states[action[0]] -= 1

                #print('total_reward', total_reward)
                self.all_capacilitys[state_index, :] = np.array(self.capacility_states)


                get_state_start_time = time.time()
                self.id_actions_dict[user_id] = action

                next_state = copy.deepcopy(self.user_states[state_index])
                next_state.extend(self.capacility_states)
                next_state.extend(self.es[state_index])
                g_at = 1
                next_state = np.array(next_state)
                get_state_end_time = time.time()
                # print('get_state_time',get_state_end_time-get_state_start_time)
                reward = g_at - 0.9 * np.max(self.relu(-1*next_state[3:6]))
                print(type(action))
                print(action)
                # if next_state[5]>0:
                #     reward = g_at + 0.5*state[action[0]] - 0.5*np.log10(next_state[5])

                # else:
                #     reward = g_at + 0.5*state[action[0]] - 0.5*np.log10(np.max(next_state[3:6]))

                total_reward+=reward
                if all_steps%1000 == 0:
                    is_save = True;step = all_steps
                else:
                    is_save = False;step = None
                # next_state[3:6] = np.log10(next_state[3:6])
                agent.percieve(state, action, reward, next_state, is_save,step)

                state = next_state.tolist()
                state_index += 1
                if state_index >= len(self.user_states):
                    break;
                state_index = state_index % len(self.user_states)
                # endtime = time.time()
                # print(endtime-starttime)
                print(episode)

            # array_start_time = time.time()
            user_state_writer = np.array(self.user_states)
            action_writer = np.array(self.actions)
            capacilitys_writer = np.array(self.all_capacilitys)
            reject_writer = np.array(self.reject)
            # array_end_time = time.time()
            # print('array time', array_end_time - array_start_time)

            # concat_start_time = time.time()
            user_action = np.concatenate((user_state_writer, action_writer.reshape([len(action_writer), 1])),
                                         axis=1)
            user_action = np.concatenate((user_action, reject_writer.reshape([len(reject_writer), 1])), axis=1)
            user_action = np.concatenate((user_action, capacilitys_writer), axis=1)
            user_action = np.concatenate((user_action, np.array(self.es).reshape([len(self.es), 1])), axis=1)
            # concat_end_time = time.time()
            # print('concat_time', concat_end_time - concat_start_time)
            user_action = user_action[np.array(self.inouts) == 1,:]
            #print('start write user-action')
            file = open('user_action_no_gt_user_state_1_1_1_logc.txt', 'w+')
            for i in range(len(user_action)):
                file.write(str(user_action[i])+'\n')
                # file.write('\n')
            file.close()
            break;
            #print('end write user_action')
            # print(self.actions)
        # file = open('user_action_gt.txt', 'w+')
        # for i in range(len(user_action)):
        #     file.write(str(user_action[i]) + '\n')
        #     # file.write('\n')
        # file.close()




if __name__ == '__main__':
    devision = Devision()
    devision.get_states()
    devision.train()