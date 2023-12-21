import math
import random
import itertools
import numpy as np
from kivy.clock import Clock
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T

class Environment:
    app = None
    device = None
    critic_network = None # MADDPG, основная нейронная сеть критика
    tgtCritic_network = None # MADDPG, целевая нейронная сеть критика
    optimizerCritic = None # MADDPG, оптимизатор нейронной сети критика
    objectiveCritic = None # MADDPG, функция потерь критика
    experience_buffer = None # MADDPG, буфер воспроизведения
    last_reward = dict()
    action_space = [0, 1, 2, 3, 4, 5, 6, 7] # 0 - left, 1 - right, 2 - up, 3 - down, 4 - more, 5 - less, 6 - rotate left, 7 - rotate right
    rewards = dict() # MADDPG, локальные награды от разных агентов
    actions = dict() # MADDPG, локальные действия от разных агентов
    observations = dict() # MADDPG, локальные состояния от разных агентов
    actions_next = dict() # MADDPG, новые действия от разных агентов
    observations_next = dict() # MADDPG, новые состояния от разных агентов
    Loss_History = [0]
    total_loss = [0]
    m_loss = [0]

    def __init__(self, steps_left=200, steps_learning=150, mode='DQN', app=None):
        self.steps_left = steps_left
        self.steps_learning = steps_learning
        self.global_step = 0
        self.mode = mode
        self.app = app
        self.current_state = None
        self.done = False
        self.usability_reward_mean = 0
        self.usability_reward_sum = 0
        self.usability_reward_median = 0
        self.last_usability_reward = 0
        self.usability_reward = 0

    def reset(self):
        self.current_state = None
        self.last_reward.clear()

    def just_starting(self): return self.current_state is None

    def get_state_tensor(self, agent):
        return torch.tensor([agent.widget.vect_state[1:]], device=agent.widget.device)
        # return torch.FloatTensor([agent.widget.vect_state[1:]], device=agent.widget.device)

    def get_state(self, agent): return self.get_state_tensor(agent)

    def get_obs_agent(self, agent): return agent.widget.vect_state[1:]

    def get_obs_size_agent(self, agent): return len(agent.widget.vect_state[1:])

    def get_actions(self): return self.action_space

    def action_space_sample(self): return random.choice(self.action_space)

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
                cur_reward[i] = 0 if reward == 0 else cur_reward[i] if delta > 0 else penalty if delta < 0 else 0
                # cur_reward[i] = 0 if reward==0 else reward if delta > 0 else penalty if delta < 0 else 0
                # if id==15 and i==3: print(f'cuv{i}={temp_cur_reward[i]}  cur{i}={cur_reward[i]}')
        else:
            cur_reward = [0. for _ in cur_reward]
        self.last_reward[id] = temp_cur_reward
        return cur_reward[:4], cur_reward[4]

    # action: 0 - left, 1 - right, 2 - up, 3 - down, 4 - more, 5 - less
    def get_local_reward(self, agent, action):
        cuv = agent.widget.vect_state  # 0 - self.id, 1 - self.taps, 2 - self.nx, 3 - self.ny, 4 - self.ns, 5 - self.nr
        id = cuv[0] - 1  # widjet id
        tuv = self.app.target_ui_vect[id]  # 0 - self.nx, 1 - self.ny, 2 - self.ns, 3 - self.nr
        cur_reward = 0
        if action==0 or action==1: cur_reward = 1 - abs(tuv[0] - cuv[2]); i = 0  # nx, position X = action: 0 - left, 1 - right
        elif action==2 or action==3: cur_reward = 1 - abs(tuv[1] - cuv[3]); i = 1  # ny, position Y = action: 2 - up, 3 - down
        elif action==4 or action==5: cur_reward = 1 - abs(tuv[2] - cuv[4]); i = 2  # ns, scale = action: 4 - more, 5 - less
        elif action==6 or action==7: cur_reward = 1 - abs(tuv[3] - cuv[5]); i = 3  # nr, rotate = action: 6 - rotate-, 7 - rotate+
        else: return 0
        temp_cur_reward = cur_reward
        if self.last_reward.get(id, None) is not None:
            reward = self.app.sliders_reward[i].value
            penalty_minus = self.app.sliders_reward[6].value
            delta = cur_reward - self.last_reward[id][i]
            cur_reward = 0 if reward == 0 else (1-cur_reward) if delta > 0 else -cur_reward if delta < 0 else 0
            # if id == 15 and i == 3: print(f'cuv{i}={temp_cur_reward}  cur{i}={cur_reward}')
        else:
            cur_reward = 0
            self.last_reward[id] = [0. for _ in range(5)]
        self.last_reward[id][i] = temp_cur_reward
        return cur_reward

    def get_activation_reward(self, agent):
        cuv = agent.widget.vect_state  # 0 - self.id, 1 - self.taps, 2 - self.nx, 3 - self.ny, 4 - self.ns, 5 - self.nr
        id = cuv[0] - 1  # widjet id
        cur_reward = cuv[1]
        temp_cur_reward = cur_reward
        i = 4  # activation set
        if self.last_reward.get(id, None) is not None:
            reward = self.app.sliders_reward[i].value
            delta = cur_reward - self.last_reward[id][i]
            cur_reward = 0 if reward == 0 else cur_reward if delta > 0 else 0
        else:
            cur_reward = 0
            self.last_reward[id] = [0. for _ in range(5)]
        self.last_reward[id][i] = temp_cur_reward
        return cur_reward

    # Usability model: Показатель плотности (DM), Удобочитаемость (Размер выделенного элемента, TeS), Баланс (BL)
    # Todi AUI: считывание элемента (Tread), наведение курсора (Tpointing), локальный поиск (Tlocal)
    # QUIM: Соответствие макета (LA), Видимость задач (TV), Горизонтальный баланс(BH), Вертикальный баланс (BV)
    # usability_metrics = ['DM', 'TeS', 'BL', 'Tread', 'Tpointing', 'Tlocal', 'LA', 'TV', 'BH', 'BV']
    def usability_reward_update(self, *args):
        a = [0. for _ in range(10)]; al = []; ar = []; at = []; ab = []; l =0
        n = float(len(self.app.FlyScatters))
        for s in self.app.FlyScatters:
            a[0] += s.widjet_area() # DM, i==0
            a[1] += int(s.height*(s.scale)>88 and s.width*(s.scale)>88) # TeS, i==1

            if s.center_x < self.app.window_width // 2: al.append(s.y); l = len(al) # BL, i==2
            if s.center_x > self.app.window_width // 2: ar.append(s.y); l = len(ar) # BL, i==2
            if s.center_y < self.app.window_height // 2: ab.append(s.x); l = len(ab) # BL, i==2
            if s.center_y > self.app.window_height // 2: at.append(s.x); l = len(at) # BL, i==2

            cuv = s.vect_state  # LA, i==6, current
            tuv = self.app.target_ui_vect[cuv[0] - 1]  # LA, i==6,  target
            a[3] += min(1, cuv[1]*10) # TR, i==3
            a[4] += np.sqrt(np.power(self.app.window_width-s.x, 2)+np.power(s.y,2))
            a[5] += l
            a[6] += 1-abs(tuv[0] - cuv[2])  # LA, i==6,  nx
            a[6] += 1-abs(tuv[1] - cuv[3])  # LA, i==6,  ny
            a[6] += 1-abs(tuv[2] - cuv[4])  # LA, i==6,  ns
            a[6] += 1-abs(tuv[3] - cuv[5])  # LA, i==6,  nr

            a[7] += 1 if s.scale > 0.9 else 0  # TV, i==7
        # usability metrics
        us_reward = [0. for _ in range(10)]
        for i in range(10):
            r_i = self.app.sliders_reward[i+7].value
            if r_i == 0: continue
            if i==0: # DM=1-1/aframe*sum_n(ai), ai and aframe represent the area of object i and the area of the frame respectively
                us_reward[i] = 1-a[i]/self.app.frame_area # sum(ai)/self.app.frame_area
            elif i==1: # TeS=1/n * sum_n(a_i), a_i=1, если площадь объекта i больше или равна 44pt х 44pt
                us_reward[i] = a[i] / n
            elif i==2: # BL=1-(|BL_vert|+|BL_hor|)/2, BL_vert=(W_L+W_R)/max(|W_L|, | W_R |), BL_hor=(W_T+W_B)/max(|W_T|,|W_B|)
                W_L = np.median(al)/self.app.window_height
                W_R = np.median(ar)/self.app.window_height
                W_T = np.median(at)/self.app.window_width
                W_B = np.median(ab)/self.app.window_width
                BL_hor = (W_L - W_R)/max(abs(W_L), abs(W_R))
                BL_vert = (W_T - W_B)/max(abs(W_T), abs(W_B))
                us_reward[i] = 1-(abs(BL_vert)+abs(BL_hor))/2
                if self.app.sliders_reward[8 + 7].value!=0: us_reward[8] = 2 * min(W_L,W_R)/(W_L+W_R) # BH=200∙W_1/(W_1+W_2 )
                if self.app.sliders_reward[9 + 7].value!=0: us_reward[9] = 2 * min(W_T,W_B)/(W_T+W_B) # BL=200∙W_1/(W_1+W_2 )
            elif i==3: # TR(i_l)=δ/(1+B(i_l)), B(i_l) - activation level i element in l position
                us_reward[i] = 1/(1+ (a[i]/n))
            elif i==4: # TP(i_l )=a+b∙log(1+i_l)
                us_reward[i] = 3.4+4.8*np.log((a[i]/n/self.app.frame_diagonal))
            elif i==5: # TL(i_l )=T_c+δ∙N_local+T_trail
                us_reward[i] = 0.01+0.1*a[i]/(n+1)+0.02
            elif i==6: # LA=100∙C_optimal/C_designed
                us_reward[i] = (a[i]/4) / n
            elif i==7:  # TV=100∙V_i/S_total,∀i, Vi – видимость функции [0, 1].
                us_reward[i] = a[i] / n
            #elif i==2: # BI=1/n*sum_n(Built_in_Icon(i)), Built_in_Icon(i)=1 if widjet (i) has icon
            #    us_reward[i] = float(1)

        self.usability_reward_mean = np.mean(us_reward)
        self.usability_reward_median = np.median(us_reward)
        self.usability_reward_sum = sum(us_reward)

        cur_reward = self.usability_reward_mean
        delta = cur_reward - self.last_usability_reward if self.last_usability_reward!=0 else 0
        self.usability_reward = 0 if delta == 0 else cur_reward if delta > 0 else -cur_reward if delta < 0 else 0
        self.last_usability_reward = cur_reward
        print(self.usability_reward, us_reward)
        return us_reward

    def take_action(self, action, agent, tensor=False):
        penalty = agent.widget.change_pos_size(action) #.data.item())
        self.steps_left -= 1
        if self.is_done(): self.done = True
        r_us = self.usability_reward
        r_pos = self.get_local_reward(agent, action)
        r_taps = self.get_activation_reward(agent)
        reward = r_pos + r_taps + r_us + penalty
        if tensor: return reward, torch.tensor([reward], device=agent.device), self.done
        return reward, self.done

    def is_done(self): return self.steps_left <= 0 or self.done

    def is_learning_MADDPG(self):
        return (self.global_step % self.steps_train == 0) and (self.global_step > self.start_steps)

    def num_actions_available(self): return len(self.get_actions())

    def num_state_available(self, agent): return len(agent.widget.vect_state)

    # Создаем минивыборку определенного объема из буфера воспроизведения
    def sample_from_expbuf(self, experience_buffer=None, batch_size=32):
        if experience_buffer == None: experience_buffer = self.experience_buffer
        # Функция возвращает случайную последовательность заданной длины
        perm_batch = np.random.permutation(len(experience_buffer))[:batch_size]
        # Минивыборка
        experience = np.array(experience_buffer)[perm_batch]
        # Возвращаем значения минивыборки по частям
        return experience[:,0], experience[:,1], experience[:,2], experience[:,3], experience[:,4], experience[:,5]

    # Выбираем возможное действие с максимальным из стратегии действий
    # с учетом дополнительного случайного шума
    def select_actionFox(self, act_prob, avail_actions_ind, n_actions, noise_rate):
        p = np.random.random(1).squeeze()
        # Добавляем случайный шум к действиям для исследования
        # разных вариантов действий
        for i in range(n_actions):
            # Создаем шум заданного уровня
            noise = noise_rate * (np.random.rand())
            # Добавляем значение шума к значению вероятности выполнения действия
            act_prob[i] = act_prob[i] + noise

        # Выбираем действия в зависимости от вероятностей их выполнения
        for j in range(n_actions):
            # Выбираем случайный элемент из списка, 0 - left, 1 - right, 2 - up, 3 - down, 4 - more, 5 - less, 6 - rotate left, 7 - rotate right
            actiontemp = random.choices(['0', '1', '2', '3', '4', '5', '6', '7'],
                                         weights=[act_prob[0], act_prob[1], act_prob[2], act_prob[3], act_prob[4],
                                                  act_prob[5], act_prob[6], act_prob[7]])
            # Преобразуем тип данных
            action = int(actiontemp[0])
            # Проверяем наличие выбранного действия в списке действий
            if action in avail_actions_ind:
                return action
            else:
                act_prob[action] = 0
                return np.random.choice(avail_actions_ind)

    def MADDPG_emulation(self, *args):
        self.usability_reward_update()

        ###########_Цикл по агентам для выполнения действий в игре_########
        for s in self.app.FlyScatters:
            # Получаем состояние среды для независимого агента
            self.obs_agent = self.get_obs_agent(s.agent)
            # Конвертируем данные в тензор
            obs_agentT = torch.FloatTensor([self.obs_agent]).to(self.device)
            # Передаем состояние среды в основную нейронную сеть и получаем стратегию действий
            action_probabilitiesT = s.actor_network(obs_agentT)
            # Конвертируем данные в numpy
            action_probabilitiesT = action_probabilitiesT.to("cpu")
            action_probabilities = action_probabilitiesT.data.numpy()[0]

            # Находим возможные действия агента в данный момент времени
            # avail_actions = env.get_avail_agent_actions(agent_id)
            # avail_actions_ind = np.nonzero(avail_actions)[0]
            avail_actions_ind = s.available_actions()

            # Выбираем возможное действие агента с учетом
            # стратегии действий и уровня случайного шума
            noise_rate = self.app.strategy.get_noise_rate(self.global_step)
            action = self.select_actionFox(action_probabilities, avail_actions_ind, self.n_actions, noise_rate)

            # Собираем действия от разных агентов
            self.action = self.actionsFox = action
            self.actions[self.agent_id] = action  # if env.actions.get(self.agent_id, None) is not None:
            # Собираем локальные состояния среды от разных агентов
            env.observations[self.agent_id] = self.obs_agent
            for i in range(obs_size):
                self.observations.append(obs_agent[agent_id][i])
        ######_конец цикла по агентам для выполнения действий в игре_######

        # Передаем действия агентов в среду, получаем награду
        reward, terminated = self.take_actions(self.actions, self)

        env.reward += reward

        # Подготовляем данные для сохранения в буфере воспроизведения
        # Если эпизод не завершился, то можно найти новые действия и состояния
        if not terminated:
            # Получаем новое состояние среды для независимого агента
            self.obs_agent_next = env.get_obs_agent(self)
            # Собираем от разных агентов новые состояния
            env.observations_next[self.agent_id] = self.obs_agent_next
            # Конвертируем данные в тензор
            obs_agent_nextT = torch.FloatTensor([self.obs_agent_next]).to(device)
            # Получаем новые действия агентов для новых состояний из целевой сети исполнителя
            action_probabilitiesT = self.widjet.target_net(obs_agent_nextT)
            # Конвертируем данные в numpy
            action_probabilitiesT = action_probabilitiesT.to("cpu")
            action_probabilities = action_probabilitiesT.data.numpy()[0]
            # Находим новые возможные действия агента
            avail_actions_ind = self.widget.available_actions()
            # Выбираем новые возможные действия
            action = select_actionFox(action_probabilities, avail_actions_ind, self.n_actions, noise_rate)
            # Собираем новые действия от разных агентов
            env.actions_next[self.agent_id] = action
        elif terminated:
            # если эпизод на этом шаге завершился, то новых действий не будет
            env.actions_next[self.agent_id] = env.actions[self.agent_id]
            env.actions_next[self.agent_id] = env.observations[self.agent_id]

        # MADDPG, работа буфера воспроизведения
        actions = list(itertools.chain(*self.actions))
        observations = list(itertools.chain(*self.observations))
        actions_next = list(itertools.chain(*self.actions_next))
        observations_next = list(itertools.chain(*self.observations_next))
        reward = list(itertools.chain(*self.rewards))
        terminated = self.is_done()

        # Сохраняем переход в буфере воспроизведения
        self.experience_buffer.append([observations, actions, observations_next, actions_next, reward, terminated])

        # Если буфер воспроизведения наполнен, начинаем обучать сеть
        ########################_начало if обучения_#######################
        if (self.global_step % self.app.steps_train == 0) and (self.global_step > self.app.start_steps):
            # Получаем минивыборку из буфера воспроизведения
            exp_obs, exp_acts, exp_next_obs, exp_next_acts, exp_rew, exp_termd = self.sample_from_expbuf(self.experience_buffer,
                                                                                                    self.app.batch_size)

            # Конвертируем данные в тензор
            exp_obs = [x for x in exp_obs]
            obs_agentsT = torch.FloatTensor([exp_obs]).to(self.device)
            exp_acts = [x for x in exp_acts]
            act_agentsT = torch.FloatTensor([exp_acts]).to(self.device)

            ###############_Обучаем нейронную сеть критика_################

            # Получаем значения из основной сети критика
            action_probabilitieQT = self.critic_network(obs_agentsT, act_agentsT)
            action_probabilitieQT = action_probabilitieQT.to("cpu")

            # Конвертируем данные в тензор
            exp_next_obs = [x for x in exp_next_obs]
            obs_agents_nextT = torch.FloatTensor([exp_next_obs]).to(self.device)
            exp_next_acts = [x for x in exp_next_acts]
            act_agents_nextT = torch.FloatTensor([exp_next_acts]).to(self.device)

            # Получаем значения из целевой сети критика
            action_probabilitieQ_nextT = self.tgtCritic_network(obs_agents_nextT, act_agents_nextT)
            action_probabilitieQ_nextT = action_probabilitieQ_nextT.to("cpu")
            action_probabilitieQ_next = action_probabilitieQ_nextT.data.numpy()[0]

            # Переформатируем y_batch размером batch_size
            y_batch = np.zeros([self.app.batch_size])
            action_probabilitieQBT = torch.empty(1, self.app.batch_size, dtype=torch.float)

            for i in range(self.app.batch_size):
                # Вычисляем целевое значение y
                y_batch[i] = exp_rew[i] + (self.app.gamma * action_probabilitieQ_next[i]) * (1 - exp_termd[i])
                action_probabilitieQBT[0][i] = action_probabilitieQT[0][i]

            y_batchT = torch.FloatTensor([y_batch])

            # Обнуляем градиенты
            self.optimizerCritic.zero_grad()

            # Вычисляем функцию потерь критика
            loss_t_critic = self.objectiveCritic(action_probabilitieQBT, y_batchT)

            # Сохраняем данные для графиков
            Loss_History.append(loss_t_critic)
            loss_n_critic = loss_t_critic.data.numpy()
            total_loss.append(loss_n_critic)
            m_loss.append(np.mean(total_loss[-1000:]))

            # Выполняем обратное распространение ошибки для критика
            loss_t_critic.backward()

            # Выполняем оптимизацию нейронной сети критика
            self.optimizerCritic.step()
            ###################_Закончили обучать критика_#################

            ##############_Обучаем нейронные сети исполнителей_############
            # Разбираем совместное состояние на локальные состояния
            obs_local1 = np.zeros([self.app.batch_size, obs_size])
            # obs_local2 = np.zeros([self.app.batch_size, obs_size])
            # obs_local3 = np.zeros([self.app.batch_size, obs_size])
            for i in range(self.app.batch_size):
                for j in range(obs_size):
                    obs_local1[i][j] = exp_obs[i][j]
            # for i in range(self.app.batch_size):
            #     k = 0
            #     for j in range(obs_size, obs_size * 2):
            #         obs_local2[i][k] = exp_obs[i][j]
            #         k = k + 1
            # for i in range(self.app.batch_size):
            #     k = 0
            #     for j in range(obs_size * 2, obs_size * 3):
            #         obs_local3[i][k] = exp_obs[i][j]
            #         k = k + 1
            # Конвертируем данные в тензор
            obs_agentT1 = torch.FloatTensor([obs_local1]).to(self.device)
            # obs_agentT2 = torch.FloatTensor([obs_local2]).to(self.device)
            # obs_agentT3 = torch.FloatTensor([obs_local3]).to(self.device)

            # Обнуляем градиенты
            self.widget.optimizer.zero_grad()
            # optimizerActor_list[1].zero_grad()
            # optimizerActor_list[2].zero_grad()

            # Подаем в нейронные сети исполнителей локальные состояния
            action_probabilitiesT1 = self.widget.actor_network(obs_agentT1)
            # action_probabilitiesT2 = actor_network_list[1](obs_agentT2)
            # action_probabilitiesT3 = actor_network_list[2](obs_agentT3)

            # Конвертируем данные в numpy
            action_probabilitiesT1 = action_probabilitiesT1.to("cpu")
            # action_probabilitiesT2 = action_probabilitiesT2.to("cpu")
            # action_probabilitiesT3 = action_probabilitiesT3.to("cpu")
            action_probabilities1 = action_probabilitiesT1.data.numpy()[0]
            # action_probabilities2 = action_probabilitiesT2.data.numpy()[0]
            # action_probabilities3 = action_probabilitiesT3.data.numpy()[0]

            # Вычисляем максимальные значения с учетом объема минивыборки
            act_full = np.zeros([env.app.batch_size, 40])  # n_agents])
            for i in range(env.app.batch_size):
                act_full[i][0] = np.argmax(action_probabilities1[i])
                # act_full[i][1] = np.argmax(action_probabilities2[i])
                # act_full[i][2] = np.argmax(action_probabilities3[i])
            act_fullT = torch.FloatTensor([act_full]).to(self.device)

            # Конвертируем данные в тензор
            exp_obs = [x for x in exp_obs]
            obs_agentsT = torch.FloatTensor([exp_obs]).to(self.device)

            # Задаем значение функции потерь для нейронных сетей исполнителей
            # как отрицательный выход критика
            actor_lossT = -env.critic_network(obs_agentsT, act_fullT)

            # Усредняем значение по количеству элементов минивыборки
            actor_lossT = actor_lossT.mean()

            # Выполняем обратное распространение ошибки
            actor_lossT.backward()

            # Выполняем оптимизацию нейронных сетей исполнителей
            self.widget.optimizer.step()
            # optimizerActor_list[1].step()
            # optimizerActor_list[2].step()

            # Собираем данные для графиков
            actor_lossT = actor_lossT.to("cpu")
            self.Loss_History_actor.append(actor_lossT)
            actor_lossN = actor_lossT.data.numpy()
            self.total_loss_actor.append(actor_lossN)
            self.m_loss_actor.append(np.mean(total_loss_actor[-1000:]))
            ##############_Закончили обучать исполнителей_#################

            # Рализуем механизм мягкой замены
            # Обновляем целевую сеть критика
            for target_param, param in zip(self.tgtCritic_network.parameters(), self.critic_network.parameters()):
                target_param.data.copy_((1 - self.app.TAU) * param.data + self.app.TAU * target_param.data)
            # Обновляем целевые сети акторов
            for target_param, param in zip(self.widjet.target_net.parameters(),
                                           self.widget.actor_network.parameters()):
                target_param.data.copy_((1 - self.app.tau) * param.data + self.app.TAU * target_param.data)

            ######################_конец if обучения_######################

        self.global_step += 1

    def set_emulation(self, on=False):
        method = self.usability_reward_update if self.mode == 'DQN' else self.MADDPG_emulation
        time = 1. / 2. if self.mode=='DQN' else 1. / 30.
        if on:
            Clock.schedule_interval(method, time)
            return True
        else:
            Clock.unschedule(method)
            return False


class Agent3:
    loss_data = [0]
    m_loss = [0]
    widget=None

    def __init__(self, strategy, memory, n_actions, widget=None):
        self.current_step = 0
        self.strategy = strategy
        self.memory = memory
        self.n_actions = n_actions # Определяем выходной размер нейронной сети
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

class Agent4:
    Loss_History_actor = [0]
    total_loss_actor = [0]
    m_loss_actor = [0]
    widget=None
    device=None

    # Храним историю действий один шаг для разных агентов
    actionsFox = 0 #np.zeros([n_agents])
    # Храним историю состояний среды один шаг для разных агентов
    obs_agent = [] #np.zeros([n_agents], dtype=object)
    obs_agent_next = [] #np.zeros([n_agents], dtype=object)

    def __init__(self, strategy, n_actions, widget=None, device=None):
        self.current_step = 0
        self.strategy = strategy
        self.n_actions = n_actions
        self.total_reward = 0.0
        self.widget = widget
        self.device = device
        self.action = 0
        self.agent_id = int(widget.id) - 1

    def step(self, env):
        # Получаем состояние среды для независимого агента
        self.obs_agent = env.get_obs_agent(self)
        # Конвертируем данные в тензор
        obs_agentT = torch.FloatTensor([self.obs_agent]).to(self.device)
        # Передаем состояние среды в основную нейронную сеть и получаем стратегию действий
        action_probabilitiesT = self.widget.actor_network(obs_agentT)
        # Конвертируем данные в numpy
        action_probabilitiesT = action_probabilitiesT.to("cpu")
        action_probabilities = action_probabilitiesT.data.numpy()[0]

        # Находим возможные действия агента в данный момент времени
        # avail_actions = env.get_avail_agent_actions(agent_id)
        # avail_actions_ind = np.nonzero(avail_actions)[0]
        avail_actions_ind = self.widget.available_actions()

        # Выбираем возможное действие агента с учетом
        # стратегии действий и уровня случайного шума
        noise_rate = self.strategy.get_noise_rate(self.current_step)
        action = select_actionFox(action_probabilities, avail_actions_ind, self.n_actions, noise_rate)

        # Собираем действия от разных агентов
        self.action = self.actionsFox = action
        env.actions[self.agent_id] = action #if env.actions.get(self.agent_id, None) is not None:
        # Собираем локальные состояния среды от разных агентов
        env.observations[self.agent_id] = self.obs_agent

        # Передаем действия агентов в среду, получаем награду
        reward, terminated = env.take_action(action, self)

        env.reward += reward

        # Подготовляем данные для сохранения в буфере воспроизведения
        # Если эпизод не завершился, то можно найти новые действия и состояния
        if not terminated:
            # Получаем новое состояние среды для независимого агента
            self.obs_agent_next = env.get_obs_agent(self)
            # Собираем от разных агентов новые состояния
            env.observations_next[self.agent_id] = self.obs_agent_next
            # Конвертируем данные в тензор
            obs_agent_nextT = torch.FloatTensor([self.obs_agent_next]).to(device)
            # Получаем новые действия агентов для новых состояний из целевой сети исполнителя
            action_probabilitiesT = self.widjet.target_net(obs_agent_nextT)
            # Конвертируем данные в numpy
            action_probabilitiesT = action_probabilitiesT.to("cpu")
            action_probabilities = action_probabilitiesT.data.numpy()[0]
            # Находим новые возможные действия агента
            avail_actions_ind = self.widget.available_actions()
            # Выбираем новые возможные действия
            action = select_actionFox(action_probabilities, avail_actions_ind, self.n_actions, noise_rate)
            # Собираем новые действия от разных агентов
            env.actions_next[self.agent_id] = action
        elif terminated:
            # если эпизод на этом шаге завершился, то новых действий не будет
            env.actions_next[self.agent_id] = env.actions[self.agent_id]
            env.actions_next[self.agent_id] = env.observations[self.agent_id]

        # Сохраняем переход в буфере воспроизведения
        env.experience_buffer.append([observations, actions, observations_next, actions_next, reward, terminated])

        # Если буфер воспроизведения наполнен, начинаем обучать сеть
        ########################_начало if обучения_#######################
        if (self.current_step % env.app.steps_train == 0) and (self.current_step > env.app.start_steps) and e.steps_learning>0:
            # Получаем минивыборку из буфера воспроизведения
            exp_obs, exp_acts, exp_next_obs, exp_next_acts, exp_rew, exp_termd = self.sample_from_expbuf(
                env.experience_buffer,
                env.app.batch_size)

            # Конвертируем данные в тензор
            exp_obs = [x for x in exp_obs]
            obs_agentsT = torch.FloatTensor([exp_obs]).to(self.device)
            exp_acts = [x for x in exp_acts]
            act_agentsT = torch.FloatTensor([exp_acts]).to(self.device)

            # Конвертируем данные в тензор
            exp_next_obs = [x for x in exp_next_obs]
            obs_agents_nextT = torch.FloatTensor([exp_next_obs]).to(self.device)
            exp_next_acts = [x for x in exp_next_acts]
            act_agents_nextT = torch.FloatTensor([exp_next_acts]).to(self.device)

            ##############_Обучаем нейронные сети исполнителей_############
            # Разбираем совместное состояние на локальные состояния
            obs_local1 = np.zeros([self.app.batch_size, obs_size])
            # obs_local2 = np.zeros([self.app.batch_size, obs_size])
            # obs_local3 = np.zeros([self.app.batch_size, obs_size])
            for i in range(self.app.batch_size):
                for j in range(obs_size):
                    obs_local1[i][j] = exp_obs[i][j]
            # for i in range(self.app.batch_size):
            #     k = 0
            #     for j in range(obs_size, obs_size * 2):
            #         obs_local2[i][k] = exp_obs[i][j]
            #         k = k + 1
            # for i in range(self.app.batch_size):
            #     k = 0
            #     for j in range(obs_size * 2, obs_size * 3):
            #         obs_local3[i][k] = exp_obs[i][j]
            #         k = k + 1
            # Конвертируем данные в тензор
            obs_agentT1 = torch.FloatTensor([obs_local1]).to(self.device)
            # obs_agentT2 = torch.FloatTensor([obs_local2]).to(self.device)
            # obs_agentT3 = torch.FloatTensor([obs_local3]).to(self.device)

            # Обнуляем градиенты
            self.widget.optimizer.zero_grad()
            # optimizerActor_list[1].zero_grad()
            # optimizerActor_list[2].zero_grad()

            # Подаем в нейронные сети исполнителей локальные состояния
            action_probabilitiesT1 = self.widget.actor_network(obs_agentT1)
            # action_probabilitiesT2 = actor_network_list[1](obs_agentT2)
            # action_probabilitiesT3 = actor_network_list[2](obs_agentT3)

            # Конвертируем данные в numpy
            action_probabilitiesT1 = action_probabilitiesT1.to("cpu")
            # action_probabilitiesT2 = action_probabilitiesT2.to("cpu")
            # action_probabilitiesT3 = action_probabilitiesT3.to("cpu")
            action_probabilities1 = action_probabilitiesT1.data.numpy()[0]
            # action_probabilities2 = action_probabilitiesT2.data.numpy()[0]
            # action_probabilities3 = action_probabilitiesT3.data.numpy()[0]

            # Вычисляем максимальные значения с учетом объема минивыборки
            act_full = np.zeros([env.app.batch_size, 40]) #n_agents])
            for i in range(env.app.batch_size):
                act_full[i][0] = np.argmax(action_probabilities1[i])
                # act_full[i][1] = np.argmax(action_probabilities2[i])
                # act_full[i][2] = np.argmax(action_probabilities3[i])
            act_fullT = torch.FloatTensor([act_full]).to(self.device)

            # Конвертируем данные в тензор
            exp_obs = [x for x in exp_obs]
            obs_agentsT = torch.FloatTensor([exp_obs]).to(self.device)

            # Задаем значение функции потерь для нейронных сетей исполнителей
            # как отрицательный выход критика
            actor_lossT = -env.critic_network(obs_agentsT, act_fullT)

            # Усредняем значение по количеству элементов минивыборки
            actor_lossT = actor_lossT.mean()

            # Выполняем обратное распространение ошибки
            actor_lossT.backward()

            # Выполняем оптимизацию нейронных сетей исполнителей
            self.widget.optimizer.step()
            # optimizerActor_list[1].step()
            # optimizerActor_list[2].step()

            # Собираем данные для графиков
            actor_lossT = actor_lossT.to("cpu")
            self.Loss_History_actor.append(actor_lossT)
            actor_lossN = actor_lossT.data.numpy()
            self.total_loss_actor.append(actor_lossN)
            self.m_loss_actor.append(np.mean(total_loss_actor[-1000:]))
            ##############_Закончили обучать исполнителей_#################

            # Рализуем механизм мягкой замены
            # Обновляем целевые сети акторов
            for target_param, param in zip(self.widjet.target_net.parameters(), self.widget.actor_network.parameters()):
                target_param.data.copy_((1 - self.app.tau) * param.data + self.app.TAU * target_param.data)

            # Подсчет количества шагов обучения
            e.steps_learning -= 1

        self.current_step += 1

        # Собираем данные для графиков
        return reward

#Определяем архитектуру нейронной сети исполнителя
class MADDPG_Actor(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(MADDPG_Actor, self).__init__()
        #На вход нейронная сеть получает состояние среды для отдельного агента
        #На выходе нейронная сеть возвращает стратегию действий
        self.MADDPG_Actor = nn.Sequential(
            #Первый линейный слой обрабатывает входные данные состояния среды
            nn.Linear(obs_size, 60),
            nn.ReLU(),
            #Второй линейный слой обрабатывает внутренние данные
            nn.Linear(60, 60),
            nn.ReLU(),
            #Третий линейный слой обрабатывает внутренние данные
            nn.Linear(60, 60),
            nn.ReLU(),
            #Четвертый линейный слой обрабатывает данные для стратегии действий
            nn.Linear(60, n_actions)
            )
        #Финальный выход нерйонной сети обрабатывается функцией Tanh()
        self.tanh_layer = nn.Tanh()
    #Вначале данные x обрабатываются полносвязной сетью с функцией ReLU
    #На выходе происходит обработка функцией Tanh()
    def forward(self, x):
        #Обработка полносвязными линейными слоями
        network_out = self.MADDPG_Actor(x)
        #Обработка функцией Tanh()
        tanh_layer_out = self.tanh_layer(network_out)
        #Выход нейронной сети
        return tanh_layer_out

#Определяем архитектуру нейронной сети критика
class MADDPG_Critic(nn.Module):
    def __init__(self, full_obs_size, n_actions_agents):
        super(MADDPG_Critic, self).__init__()
        #На вход нейронная сеть получает состояние среды,
        #включающее все локальные состояния среды от отдельных агентов
        #и все выполненные действия отдельных агентов
        #На выходе нейронная сеть возвращает корректирующее значение
        self.network = nn.Sequential(
            #Первый линейный слой обрабатывает входные данные
            nn.Linear(full_obs_size+n_actions_agents, 202),
            nn.ReLU(),
            #Второй линейный слой обрабатывает внутренние данные
            nn.Linear(202, 60),
            nn.ReLU(),
            #Третий линейный слой обрабатывает внутренние данные
            nn.Linear(60, 30),
            nn.ReLU(),
            #Четвертый линейный слой обрабатывает выходные данные
            nn.Linear(30, 1)
            )
    #Данные x последовательно обрабатываются полносвязной сетью с функцией ReLU
    def forward(self, state, action):
        #Объединяем данные состояний и действий для передачи в сеть
        x = torch.cat([state, action], dim=2)
        #Результаты обработки
        Q_value = self.network(x)
        #Финальный выход нейронной сети
        return Q_value

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

class NoiseRateStrategy():
    def __init__(self, noise_rate_min, noise_rate_max, noise_decay_steps):
        self.noise_rate_min = noise_rate_min
        self.noise_rate_max = noise_rate_max
        self.noise_decay_steps = noise_decay_steps

    def get_noise_rate(self, current_step):
        noise_rate = max(noise_rate_min, noise_rate_max - (noise_rate_max - noise_rate_min) * current_step / noise_decay_steps)
        return noise_rate

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
def get_optimizer_Adam(policy_net, lr):
    return optim.Adam(params=policy_net.parameters(), lr=lr)

def get_MSELoss_func():
    return nn.MSELoss()

def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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