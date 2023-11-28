import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T

class Environment:
    app = None
    last_reward = dict()
    action_space = [0, 1, 2, 3, 4, 5, 6, 7] # 0 - left, 1 - right, 2 - up, 3 - down, 4 - more, 5 - less, 6 - rotate left, 7 - rotate right

    def __init__(self, steps_left=200, steps_learning=150, app=None):
        self.steps_left = steps_left
        self.steps_learning = steps_learning
        self.app = app
        self.current_state = None
        self.done = False

    def reset(self):
        self.current_state = None
        self.last_reward.clear()

    def just_starting(self):
        return self.current_state is None

    def get_state_tensor2(self, agent):
        return torch.FloatTensor([agent.widget.vect_state[1:]], device=agent.widget.device)

    def get_state_tensor(self, agent):
        return torch.tensor([agent.widget.vect_state[1:]], device=agent.widget.device)

    def get_state(self, agent):
        return self.get_state_tensor(agent)

    def get_obs_agent(self, agent): # Agent2
        return agent.widget.vect_state[1:]

    def get_actions(self):
        return self.action_space

    def action_space_sample(self):
        return random.choice(self.action_space)

    # Usability model: Показатель плотности (DM), Удобочитаемость (Размер выделенного элемента, TeS), Баланс (BL)
    # Todi AUI: считывание элемента (Tread), наведение курсора (Tpointing), локальный поиск (Tlocal)
    # QUIM: Соответствие макета (LA), Видимость задач (TV), Горизонтальный баланс(BH), Вертикальный баланс (BV)
    # usability_metrics = ['DM', 'TeS', 'BL', 'Tread', 'Tpointing', 'Tlocal', 'LA', 'TV', 'BH', 'BV']
    def get_rewards(self, agent):
        cuv = agent.widget.vect_state #0 - self.id, 1 - self.taps, 2 - self.nx, 3 - self.ny, 4 - self.ns, 5 - self.nr
        id = cuv[0] - 1 # widjet id
        tuv = self.app.target_ui_vect[id] # 0 - self.nx, 1 - self.ny, 2 - self.ns, 3 - self.nr
        cur_reward = []
        cur_reward.append(1 - abs(tuv[0] - cuv[2]))  # nx, position X
        cur_reward.append(1 - abs(tuv[1] - cuv[3]))  # ny, position Y
        cur_reward.append(1 - abs(tuv[2] - cuv[4]))  # ns, scale
        cur_reward.append(1 - abs(tuv[3] - cuv[5]))  # nr, rotate
        cur_reward.append(min(1, cuv[1] / 10.))  # taps
        temp_cur_reward = cur_reward.copy()
        if self.last_reward.get(id, None) is not None:
            # temp_last_reward = self.last_reward[id].copy()
            delta_cur_last_reward = [cur_reward[i]-self.last_reward[id][i] for i in range(len(self.last_reward[id]))]
            penalty = self.app.sliders_reward[6].value
            # local reward
            for i in range(5):
                reward = self.app.sliders_reward[i].value
                delta = delta_cur_last_reward[i]
                cur_reward[i] = 0 if reward==0 else reward if delta > 0 else penalty if delta < 0 else 0
                if id==0 and i==3:# and cur_reward[i]!=0 and cur_reward[i]!=-0.95 and cur_reward[i]!=0.55:
                    print(f'cuv{i}={temp_cur_reward[i]}  cur{i}={cur_reward[i]}')
            # usability metrics
            for i in range(7,len(self.app.sliders_reward)):
                if self.app.sliders_reward[i].value==0: continue
                pass
        else:
            cur_reward = [0. for _ in cur_reward]
        self.last_reward[id] = temp_cur_reward
        return cur_reward[:4], cur_reward[4]

    def take_action(self, action, agent, tensor=False):
        penalty = agent.widget.change_pos_size(action) #.data.item())
        self.steps_left -= 1
        if self.is_done(): self.done = True
        r_pos, r_taps = self.get_rewards(agent)
        reward = sum(r_pos) + r_taps + penalty
        #reward = sum(r_pos)/len(r_pos) + r_taps + penalty
        terminated = True if self.steps_left==0 else False
        if tensor: return reward, torch.tensor([reward], device=agent.device), terminated
        return reward, terminated
        #rewards.sort()  #reverse=True)
        #return sum(rewards)/100
        #return min(rewards[:3])
        #return sum(r_pos) / len(r_pos)  + r_taps
        #return random.choice(rewards)
        #return random.random()-0.5
        #return random.choice(self.get_rewards())

    def is_done(self):
        return self.steps_left <= 0 or self.done

    def num_actions_available(self):
        return len(self.get_actions())

    def num_state_available(self, agent):
        return len(agent.widget.vect_state)


class Agent3:
    loss_data = [0]
    m_loss = [0]
    widget=None

    def __init__(self, strategy, memory, num_actions, widget=None):
        self.current_step = 0
        self.strategy = strategy
        self.memory = memory
        self.qofa_out = num_actions # Определяем выходной размер нейронной сети
        self.total_reward = 0.0
        self.widget = widget

    # Выбираем возможное действие с максимальным Q-значением в зависимости от эпсилон
    def select_action(self, action_probability, avail_actions_ind):
        epsilon = self.strategy.get_exploration_rate2(self.current_step)
        self.current_step += 1
        # Исследуем пространство действий
        if np.random.rand() < epsilon:
            return np.random.choice(avail_actions_ind)
        else:
            action = np.argmax(action_probability)
            return action #if action in avail_actions_ind else np.random.choice(avail_actions_ind)

    def step(self, e):
        # Получаем состояние среды для независимого агента IQL
        state = e.get_obs_agent(self) #Храним историю состояний среды один шаг

        # Конвертируем данные в numpy
        action_probability = self.widget.policy_net.predict(np.array(state))

        avail_actions_ind = self.widget.available_actions()
        # Выбираем возможное действие агента с учетом
        # максимального Q-значения и параметра эпсилон
        action = self.select_action(action_probability, avail_actions_ind)

        # Передаем действия агентов в среду, получаем награду
        reward, done = e.take_action(action, self)

        # Получаем новое состояние среды
        next_state = e.get_obs_agent(self)

        # Сохраняем переход в буфере воспроизведения для каждого агента
        # self.memory.push(Experience(obs_agentT, action, rewardT, obs_agent_nextT))
        self.memory.append((state, action, reward, next_state))

        if not done and e.app.batch_size < len(self.memory) and e.steps_learning>0:
            minibatch = random.sample(list(self.memory), e.app.batch_size)

            state = np.array([i[0] for i in minibatch])
            action = [i[1] for i in minibatch]
            rewards = [i[2] for i in minibatch]
            next_state = np.array([i[3] for i in minibatch])

            q_value = self.widget.policy_net.predict(np.array(state))
            ns_model_pred = self.widget.target_net.predict(np.array(next_state))

            for i in range(0, e.app.batch_size):
                q_value[i][action[i]] = rewards[i] + e.app.gamma * np.max(ns_model_pred[i])

            loss = self.widget.policy_net.fit(state, q_value)
            self.loss_data.append(loss)
            self.m_loss.append(np.mean(self.loss_data[-1000:]))

            # Подсчет количества шагов обучения
            e.steps_learning -= 1

        # Собираем данные для графиков
        return reward

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay, decay_steps):
        self.start = start
        self.end = end
        self.decay = decay
        self.decay_steps = decay_steps

    def get_exploration_rate(self, current_step):
        epsilon = self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)
        return epsilon

    def get_exploration_rate2(self, current_step):
        epsilon = max(self.end, self.start - (self.start - self.end) * current_step / self.decay_steps)
        return epsilon

# class Agent:
#     loss_data = [0]
#     total_loss = [0]
#     m_loss = [0]
#     widget = None
#     memory = None
#     last_reward = dict()
#
#     def __init__(self, strategy, memory, num_actions, widget=None, device=None):
#         self.current_step = 0
#         self.strategy = strategy
#         self.memory = memory
#         self.num_actions = num_actions
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if not device else device
#         self.total_reward = 0.0
#         self.widget = widget
#
#     def select_action(self, state, policy_net, env):
#         sample = random.random()
#         eps_threshold = self.strategy.get_exploration_rate(self.current_step)
#         self.current_step += 1
#         if sample > eps_threshold:
#             with torch.no_grad():
#                 # t.max(1) will return the largest column value of each row.
#                 # second column on max result is index of where max element was
#                 # found, so we pick action with the larger expected reward.
#                 return policy_net(state).max(1)[1].view(1, 1)
#         else:
#             return torch.tensor([[env.action_space_sample()]], device=self.device, dtype=torch.long)
#         # if rate > random.random():
#         #     action = random.randrange(self.num_actions)
#         #     return torch.tensor([[action]]).to(self.device) #explore
#         # else:
#         #     with torch.no_grad(): #exploit
#         #         #return torch.tensor([policy_net(state).argmax(dim=1)], device=self.device)
#         #         return torch.tensor([[policy_net(state).max(1)[1].view(1, 1)]], device=self.device)
#
#     def step(self, e):
#         state = e.get_obs_agent(self)
#         state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
#         # episode_durations = []
#         # for timestep in count():
#         #     if e.done:
#         #         episode_durations.append(timestep)
#         #         plot(episode_durations, 100)
#         #         break
#
#         action = self.select_action(state, self.widget.policy_net, e)
#         r, reward, terminated = e.take_action(action.item(), self)
#         observation = e.get_obs_agent(self)
#         if terminated:
#             next_state = None
#         else:
#             next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
#         self.memory.push(state, action, next_state, reward)
#         state = next_state
#
#         if self.memory.can_provide_sample(e.app.batch_size) and e.steps_learning > 0:
#             transitions = self.memory.sample(e.app.batch_size)
#             batch = Transition(*zip(*transitions))
#             non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                                     batch.next_state)), device=self.device, dtype=torch.bool)
#             non_final_next_states = torch.cat([s for s in batch.next_state
#                                                if s is not None])
#             state_batch = torch.cat(batch.state)
#             action_batch = torch.cat(batch.action)
#             reward_batch = torch.cat(batch.reward)
#
#             # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#             # columns of actions taken. These are the actions which would've been taken
#             # for each batch state according to policy_net
#             state_action_values = self.widget.policy_net(state_batch).gather(1, action_batch)
#
#             # Compute V(s_{t+1}) for all next states.
#             # Expected values of actions for non_final_next_states are computed based
#             # on the "older" target_net; selecting their best reward with max(1)[0].
#             # This is merged based on the mask, such that we'll have either the expected
#             # state value or 0 in case the state was final.
#             next_state_values = torch.zeros(e.app.batch_size, device=self.device)
#             with torch.no_grad():
#                 next_state_values[non_final_mask] = self.widget.target_net(non_final_next_states).max(1)[0]
#             # Compute the expected Q values
#             expected_state_action_values = (next_state_values * e.app.gamma) + reward_batch
#
#             # Compute Mean Square loss
#             # loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
#             # Compute Huber loss
#             criterion = nn.SmoothL1Loss()
#             loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
#             self.loss_data.append(loss.data.item())
#             loss_n = loss.data.numpy()
#             self.total_loss.append(loss_n)
#             self.m_loss.append(np.mean(self.total_loss[-1000:]))
#
#             # Optimize the model
#             self.widget.optimizer.zero_grad()
#             loss.backward()
#             # In-place gradient clipping
#             torch.nn.utils.clip_grad_value_(self.widget.policy_net.parameters(), 100)
#             self.widget.optimizer.step()
#
#             e.steps_learning -= 1
#
#         # self.reward_data.append(reward.data.item())
#
#         # actions = e.get_actions()
#         # action = random.choice(actions)
#         # reward, t = e.take_action(action, self)
#         # print(t)
#         return r

# class Agent2:
#     loss_data = [0]
#     total_loss = [0]
#     m_loss = [0]
#     widget = None
#
#     def __init__(self, strategy, memory, num_actions, widget=None, device=None):
#         self.current_step = 0
#         self.strategy = strategy
#         self.memory = memory
#         self.qofa_out = num_actions  # Определяем выходной размер нейронной сети
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if not device else device
#         self.total_reward = 0.0
#         self.objective = nn.SmoothL1Loss()  # Huber loss  #self.objective = nn.MSELoss()
#         self.widget = widget
#
#     # Выбираем возможное действие с максимальным Q-значением в зависимости от эпсилон
#     def select_actionFox(self, action_probabilities, avail_actions_ind):
#         epsilon = self.strategy.get_exploration_rate2(self.current_step)
#         self.current_step += 1
#         # Исследуем пространство действий
#         if np.random.rand() < epsilon:
#             return torch.tensor([np.random.choice(avail_actions_ind)]).to(self.device)
#         else:
#             # Находим возможное действие:
#             # Проверяем есть ли действие в доступных действиях агента
#             for ia in action_probabilities:
#                 action = np.argmax(action_probabilities)
#                 if action in avail_actions_ind:
#                     # return action
#                     return torch.tensor([action]).to(self.device)
#                     # with torch.no_grad():  # exploit
#                     #     # return torch.tensor([policy_net(state).argmax(dim=1)], device=self.device)
#                     #     return torch.tensor([policy_net(state).max(1)[1].view(1, 1)], device=self.device)
#                 else:
#                     action_probabilities[action] = 0
#
#     # Создаем минивыборку определенного объема из буфера воспроизведения
#     def sample_from_expbuf(self, experience_buffer, batch_size):
#         # Функция возвращает случайную последовательность заданной длины из его элементов.
#         perm_batch = np.random.permutation(len(experience_buffer))[:batch_size]
#         # Минивыборка
#         experience = np.array(experience_buffer)[perm_batch]
#         # Возвращаем значения минивыборки по частям
#         return experience[:, 0], experience[:, 1], experience[:, 2], experience[:, 3]
#
#     def step(self, e):
#         # Получаем состояние среды для независимого агента IQL
#         obs_agentT = e.get_state_tensor2(self)  # Храним историю состояний среды один шаг
#         # obs_agentT = torch.FloatTensor([self.obs_agent], device=self.device)
#
#         # Передаем состояние среды в основную нейронную сеть
#         # и получаем Q-значения для каждого действия
#         # action_probabilitiesT = self.widget.policy_net(obs_agentT).to("cpu")
#         with torch.no_grad():
#             action_probabilitiesT = self.widget.policy_net(obs_agentT)
#             action_probabilitiesT = action_probabilitiesT.to(self.device)
#         # Конвертируем данные в numpy
#         action_probabilities = action_probabilitiesT.data.numpy()[0]
#
#         avail_actions_ind = self.widget.available_actions()
#         # Выбираем возможное действие агента с учетом
#         # максимального Q-значения и параметра эпсилон
#         action = self.select_actionFox(action_probabilities, avail_actions_ind)
#
#         # Передаем действия агентов в среду, получаем награду
#         reward, rewardT, _ = self.take_action(action, self)
#
#         # Получаем новое состояние среды
#         obs_agent_nextT = e.get_state_tensor2(self)
#
#         # Сохраняем переход в буфере воспроизведения для каждого агента
#         self.memory.push(Experience(obs_agentT, action, rewardT, obs_agent_nextT))
#
#         l = 0.
#
#         # Если буфер воспроизведения наполнен, начинаем обучать сеть
#         if self.memory.can_provide_sample(e.app.batch_size) and e.steps_learning > 0:
#             # Получаем минивыборку из буфера воспроизведения
#             experiences = self.memory.sample(e.app.batch_size)
#             # Конвертируем данные состояния в тензоры
#             obs_agentT, actions, rewards, obs_agentT_next = extract_tensors(experiences)
#             # obs_agentT = torch.FloatTensor([exp_obs]).to(self.device)
#
#             # Подаем минивыборку в основную нейронную сеть чтобы получить Q(s,a)
#             action_probabilitiesT = self.widget.policy_net(obs_agentT)
#             action_probabilitiesT = action_probabilitiesT.to(self.device)
#             # action_probabilities = action_probabilitiesT.data.numpy()[0]
#
#             # Конвертируем данные след.состояния в тензор
#             # obs_agentT_next = torch.FloatTensor([exp_next_obs]).to(self.device)
#
#             # Подаем минивыборку в целевую нейронную сеть чтобы получить Q(s,a)
#             action_probabilitiesT_next = self.widget.target_net(obs_agentT_next)
#             action_probabilitiesT_next = action_probabilitiesT_next.to(self.device)
#             action_probabilities_next = action_probabilitiesT_next.data.numpy()[0]
#
#             # Вычисляем целевое значение y
#             y_batch = rewards + e.app.gamma * np.max(action_probabilities_next, axis=-1)
#             # target_q_values = (action_probabilitiesT_next * e.app.gamma) + rewards
#
#             # Переформатируем y_batch размером batch_size
#             y_batchT = y_batch.unsqueeze(1).repeat(1, self.qofa_out)
#             # y_batch64 = np.zeros([e.app.batch_size, self.qofa_out])
#             # for i in range(e.app.batch_size):
#             #     for j in range(self.qofa_out):
#             #         y_batch64[i][j] = y_batch[i]
#             # # Конвертируем данные в тензор
#             # #y_batchT = torch.FloatTensor([y_batch64])
#             # y_batchT = torch.from_numpy(y_batch64)
#
#             # Обнуляем градиенты
#             self.widget.optimizer.zero_grad()
#
#             # Вычисляем функцию потерь
#             loss_t = self.objective(action_probabilitiesT, y_batchT)
#             # cl = action_probabilitiesT.max(dim=1)[0].detach().unsqueeze(1)
#             # tl = target_q_values.unsqueeze(1)
#             # loss_t = self.objective(cl, tl)
#
#             # Сохраняем данные для графиков
#             self.loss_data.append(loss_t.data.item())
#             loss_n = loss_t.data.numpy()
#             self.total_loss.append(loss_n)
#             self.m_loss.append(np.mean(self.total_loss[-1000:]))
#
#             # Выполняем обратное распространение ошибки
#             loss_t.backward()
#
#             torch.nn.utils.clip_grad_value_(self.widget.policy_net.parameters(), 100)
#             # Выполняем оптимизацию нейронных сетей
#             self.widget.optimizer.step()
#
#             # Подсчет количества шагов обучения
#             e.steps_learning -= 1
#
#         # Собираем данные для графиков
#         return reward

# class QValues():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     @staticmethod
#     def get_current(policy_net, states, actions):
#         # st = torch.tensor([[0.0000, 0.0100, 0.8100, 1.0100, 0.0200],
#         #                    [0.0000, 0.0100, 0.8200, 1.0100, 0.0200]], device=device)
#         # at = torch.tensor([0, 1], device=device)
#         # t = policy_net(st).gather(dim=0, index=at.unsqueeze(-1))
#         return policy_net(states).gather(dim=0, index=actions.unsqueeze(-1))
#
#     @staticmethod
#     def get_next(target_net, next_states):
#         final_state_locations = next_states.flatten(start_dim=1) \
#             .max(dim=1)[0].eq(0).type(torch.bool)
#         non_final_state_locations = (final_state_locations == False)
#         non_final_states = next_states[non_final_state_locations]
#         batch_size = next_states.shape[0]
#         values = torch.zeros(batch_size).to(QValues.device)
#         values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
#         return values
#
#     @staticmethod
#     def get_next_v2(target_net, next_states):
#         # batch_size = next_states.shape[0]
#         # values = torch.zeros(batch_size).to(QValues.device)
#         # values[non_final_state_locations] = target_net(next_states).max(dim=1)[0].detach()
#         return target_net(next_states).max(dim=1)[0].detach()


# def extract_tensors(experiences):
#     batch = Experience(*zip(*experiences))
#     t1 = torch.cat(batch.state)
#     t2 = torch.cat(batch.action)
#     t3 = torch.cat(batch.reward)
#     t4 = torch.cat(batch.next_state)
#     return (t1, t2, t3, t4)
#
# def get_optimizer_AdamW(policy_net, lr):
#     return optim.AdamW(params=policy_net.parameters(), lr=lr, amsgrad=True)
#
# def get_optimizer_Adam(policy_net, lr):
#     return optim.Adam(params=policy_net.parameters(), lr=lr)
#
# def get_nn_module(input_length, device):
#     return DQN(input_length).to(device)
#
# def get_nn_module1(input_length, device):
#     return DQNtorch(input_length).to(device)
#
# def get_nn_module2(input_length, device):
#     return Q_network(input_length).to(device)

# class DQN(nn.Module):
#     def __init__(self, state_len):
#         super(DQN, self).__init__()
#         # 0 - left, 1 - right, 2 - up, 3 - down, 4 - more, 5 - less, 6 - rotate left, 7 - rotate right
#         self.fc1 = nn.Linear(in_features=state_len, out_features=64)
#         self.fc2 = nn.Linear(in_features=64, out_features=64)
#         self.out = nn.Linear(in_features=64, out_features=8)
#         self.sm_layer = nn.Softmax(dim=1)
#         # self.fc1 = nn.Linear(in_features=state_len, out_features=40)
#         # self.fc2 = nn.Linear(in_features=40, out_features=64)
#         # self.out = nn.Linear(in_features=64, out_features=8)
#
#     def forward(self, t):
#         # t = t.flatten(start_dim=1) #for image processing
#         t = F.relu(self.fc1(t))
#         t = F.relu(self.fc2(t))
#         return self.sm_layer(self.out(t))


# class Q_network(nn.Module):
#     def __init__(self, obs_size, n_actions=8):
#         super(Q_network, self).__init__()
#         self.Q_network = nn.Sequential(
#             nn.Linear(obs_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, n_actions)
#         )
#         self.sm_layer = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         q_network_out = self.Q_network(x)
#         sm_layer_out = self.sm_layer(q_network_out)
#         return sm_layer_out

# class DQNtorch(nn.Module):
#
#     def __init__(self, n_observations, n_actions=8):
#         super(DQNtorch, self).__init__()
#         self.layer1 = nn.Linear(n_observations, 128)
#         self.layer2 = nn.Linear(128, 128)
#         self.layer3 = nn.Linear(128, n_actions)
#         self.sm_layer = nn.Softmax(dim=1)
#
#     # Called with either one element to determine next action, or a batch
#     # during optimization. Returns tensor([[left0exp,right0exp]...]).
#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         return self.sm_layer(self.layer3(x))


# PyTorch
# Transition = namedtuple(
#     'Transition',
#     ('state', 'action', 'next_state', 'reward')
# )
#
# Experience = namedtuple(
#     'Experience',
#     ('state', 'action', 'reward', 'next_state')
# )


# class ReplayMemory():
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.push_count = 0
#
#     def push(self, experience):
#         if len(self.memory) < self.capacity:
#             self.memory.append(experience)
#         else:
#             self.memory[self.push_count % self.capacity] = experience
#         self.push_count += 1
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def can_provide_sample(self, batch_size):
#         return len(self.memory) >= batch_size
#
#
# class ReplayMemoryPyTorch(object):
#
#     def __init__(self, capacity):
#         self.memory = deque([], maxlen=capacity)
#
#     def push(self, *args):
#         self.memory.append(Transition(*args))
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)
#
#     def can_provide_sample(self, batch_size):
#         return len(self.memory) >= batch_size
#
#
# def plot(values, moving_avg_period):
#     plt.figure(2)
#     plt.clf()
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#     plt.plot(values)
#     moving_avg = get_moving_average(moving_avg_period, values)
#     plt.plot(moving_avg)
#     plt.pause(0.001)
#     print("Episode", len(values), '\n', moving_avg_period, 'episode moving avg:', moving_avg[-1])
#
# def get_moving_average(period, values):
#     values = torch.tensor(values, dtype=torch.float)
#     if len(values) >= period:
#         moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
#         moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
#         return moving_avg.numpy()
#     else:
#         moving_avg = torch.zeros(len(values))
#         return moving_avg.numpy()
#
# # Learning Intrinsic Symbolic Rewards in RL (2020)
# # https://arxiv.org/pdf/2010.03694.pdf
#
# def add(left, right):
#     return left + right
#
# def subtract(left, right):
#     return left - right
#
# def multiply(left, right):
#     return left * right
#
# def cos(left):
#     return np.cos(left)
#
# def sin(left):
#     return np.sin(left)
#
# def tan(left):
#     return np.tan(left)
#
# def npmax(nums):
#     return np.maxmimum(nums)
#
# def npmin(nums):
#     return np.minimum(nums)
#
# def pass_greater(left, right):
#     if left > right: return left
#     return right
#
# def pass_smaller(left, right):
#     if left < right: return left
#     return right
#
# def equal_to(left, right):
#     return float(left == right)
#
# def gate(left, right, condtion):
#     if condtion <= 0: return left
#     else: return right
#
# def square(left):
#     return left * left
#
# def is_negative(left):
#     if left < 0: return 1.0
#     return 0.0
#
# def div_by_100(left):
#     return left / 100.0
#
# def div_by_10(left):
#     return left / 10.0
#
# def protected_div(left, right):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         x = np.divide(left, right)
#         if isinstance(x, np.ndarray):
#             x[np.isinf(x)] = 1
#             x[np.isnan(x)] = 1
#         elif np.isinf(x) or np.isnan(x):
#             x = 1
#     return x
#
# # An example of a discovered symbolic reward on PixelCopter. We unroll the correspond-
# # ing symbolic tree into Python-like code that can be parsed and debugged.
# # {si} represent state observations.
# def get_intrinsic_reward(s_0, s_1, s_2, s_3, s_4, s_5, s_6, s_7):
#     p_1 = tan(cos(s_4)); p_2 = cos(s_3); p_3 = pass_smaller(p_1, p_2)
#     x_1 = multiply(-1, abs(subtract(s_7, p_3)))
#     q_1 = multiply(-1, abs(subtract(1, s_4)))
#     q_2 = max([s_2, 1, s_7, q_1, 0])
#     q_3 = max([q_2, s_7, cos(0), multiply(s_0, s_6), multiply(s_5, subtract(s_6, 1))])
#     y_1 = div_by_10(q_3)
#     y_2 = square(s_7)
#     y_3 = protected_div(1, div_by_100(s_0))
#     x_2 = gate(y_1, y_2, y_3)
#     z = equal_to(x_2, x_1)
#     reward = add(0, pass_smaller(div_by_10(s_7), z))
#     return reward