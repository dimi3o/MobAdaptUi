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

class Environment:
    emulation = False
    app = None
    strategy = None
    device = None
    last_reward = dict()
    action_space = [0,1,2,3,4,5,6,7]  # 0 - left, 1 - right, 2 - up, 3 - down, 4 - more, 5 - less, 6 - rotate left, 7 - rotate right
    critic_network = None # MADDPG, основная нейронная сеть критика
    tgtCritic_network = None # MADDPG, целевая нейронная сеть критика
    optimizerCritic = None # MADDPG, оптимизатор нейронной сети критика
    objectiveCritic = None # MADDPG, функция потерь критика
    experience_buffer = None # MADDPG, буфер воспроизведения
    Loss_History = [0]
    Reward_History = [0]
    winrate_history = [0]
    reward_data = [0]
    total_loss = [0]
    m_loss = [0]
    Loss_History_actor = [0]
    total_loss_actor = [0]
    m_loss_actor = [0]

    def __init__(self, steps_left=200, steps_learning=150, mode='IQL', app=None):
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
        self.episode_reward = 0 # Награда за эпизод

    def reset(self):
        self.current_state = None
        self.last_reward.clear()

    def just_starting(self): return self.current_state is None

    def get_state_tensor(self, agent):
        return torch.tensor([agent.widget.vect_state[1:]], device=agent.widget.device)

    def get_state(self, agent): return self.get_state_tensor(agent)

    def get_obs_agent(self, agent): return agent.widget.vect_state[1:]

    def get_obs_size_agent(self, agent): return len(agent.widget.vect_state[1:])

    def get_actions(self): return self.action_space

    def action_space_sample(self): return random.choice(self.action_space)

    def get_rewards_v1(self, agent):
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
            delta_cur_last_reward = [cur_reward[i]-self.last_reward[id][i] for i in range(len(self.last_reward[id]))]
            penalty = self.app.sliders_reward[6].value
            # local reward
            for i in range(5):
                reward = self.app.sliders_reward[i].value
                delta = delta_cur_last_reward[i]
                cur_reward[i] = 0 if reward == 0 else cur_reward[i] if delta > 0 else penalty if delta < 0 else 0
        else:
            cur_reward = [0. for _ in cur_reward]
        self.last_reward[id] = temp_cur_reward
        return cur_reward[:4], cur_reward[4]

    # action: 0 - left, 1 - right, 2 - up, 3 - down, 4 - more, 5 - less
    def get_local_reward(self, agent, action):
        cuv = agent.widget.vect_state  # 0 - self.id, 1 - self.taps, 2 - self.nx, 3 - self.ny, 4 - self.ns, 5 - self.nr
        id = cuv[0] - 1  # widjet id
        tuv = self.app.target_ui_vect[id]  # 0 - self.nx, 1 - self.ny, 2 - self.ns, 3 - self.nr
        if action==0 or action==1: rik = 1 - abs(tuv[0] - cuv[2]); i = 0  # nx, position X = action: 0 - left, 1 - right
        elif action==2 or action==3: rik = 1 - abs(tuv[1] - cuv[3]); i = 1  # ny, position Y = action: 2 - up, 3 - down
        elif action==4 or action==5: rik = 1 - abs(tuv[2] - cuv[4]); i = 2  # ns, scale = action: 4 - more, 5 - less
        elif action==6 or action==7: rik = 1 - abs(tuv[3] - cuv[5]); i = 3  # nr, rotate = action: 6 - rotate-, 7 - rotate+
        else: return 0
        if self.last_reward.get(id, None) is not None:
            reward = self.app.sliders_reward[i].value
            penalty_minus = -rik/2 if self.app.sliders_reward[6].value!=0 else 0
            delta = rik - self.last_reward[id][i]
            cur_reward = 0 if reward == 0 else (1-rik) if delta > 0 else penalty_minus if delta < 0 else 0
        else:
            cur_reward = 0
            self.last_reward[id] = [0. for _ in range(5)]
        self.last_reward[id][i] = rik
        return cur_reward, rik

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

        self.usability_reward_mean = np.mean(us_reward)
        self.usability_reward_median = np.median(us_reward)
        self.usability_reward_sum = sum(us_reward)

        cur_reward = self.usability_reward_mean
        delta = cur_reward - self.last_usability_reward if self.last_usability_reward!=0 else 0
        self.usability_reward = 0 if delta == 0 else cur_reward if delta > 0 else -cur_reward/2 if delta < 0 else 0
        self.last_usability_reward = cur_reward
        print(self.usability_reward, us_reward)
        return us_reward

    def take_action(self, action, agent, us_r=True,tensor=False):
        penalty = agent.widget.change_pos_size(action)
        self.steps_left -= 1
        if self.is_done(): self.done = True
        r_us = self.usability_reward if us_r else 0
        r_pos, rik = self.get_local_reward(agent, action)
        r_taps = self.get_activation_reward(agent)
        reward = r_pos + r_taps + r_us + penalty
        if tensor: return reward, torch.tensor([reward], device=agent.device), self.done
        return reward, self.done, rik

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

        widjets = self.app.FlyScatters
        n_agents = len(widjets) #Количество агентов
        n_actions = len(self.action_space) #Количество действий агента

        # Обнуляем промежуточные переменные
        actions = []  # MADDPG, локальные действия от разных агентов
        observations = []
        batch_size = self.app.batch_size_MADDPG
        action = 0
        obs_size = 5
        # Храним историю действий один шаг для разных агентов
        actionsFox = np.zeros([n_agents])
        # Храним историю состояний среды один шаг для разных агентов
        obs_agent = np.zeros([n_agents], dtype=object)
        obs_agent_next = np.zeros([n_agents], dtype=object)
        # Обновляем и выводим динамический уровень случайного шума
        noise_rate = self.strategy.get_noise_rate(self.global_step)

        ###########_Цикл по агентам-виджетам для выполнения действий_########
        for w in widjets:
            agent_id = w.id-1
            # Получаем состояние среды для независимого агента
            cur_obs_agent = self.get_obs_agent(w.agent)
            obs_size = len(cur_obs_agent) # Получаем информацию о состоянии
            obs_agent[agent_id] = cur_obs_agent
            # Конвертируем данные в тензор
            obs_agentT = torch.FloatTensor([obs_agent[agent_id]]).to(self.device)
            # Передаем состояние среды в основную нейронную сеть и получаем стратегию действий
            action_probabilitiesT = w.actor_network(obs_agentT)
            # Конвертируем данные в numpy
            action_probabilitiesT = action_probabilitiesT.to("cpu")
            action_probabilities = action_probabilitiesT.data.numpy()[0]

            # Находим возможные действия агента в данный момент времени
            avail_actions_ind = w.available_actions()

            # Выбираем возможное действие агента с учетом
            # стратегии действий и уровня случайного шума
            action = self.select_actionFox(action_probabilities, avail_actions_ind, n_actions, noise_rate)

            # Собираем действия от разных агентов
            actions.append(action)
            actionsFox[agent_id] = action
            # Собираем локальные состояния среды от разных агентов
            for i in range(obs_size): observations.append(obs_agent[agent_id][i])
        ######_конец цикла по агентам для выполнения действий в игре_######

        # Передаем действия агентов в среду, получаем награду
        reward_list = np.zeros([n_agents])
        reward = 0; terminated = False
        for i, action in enumerate(actionsFox):
            r, d, rik = self.take_action(int(action), widjets[i].agent, False)
            reward_list[i] = r; terminated |= d

        # Суммируем награды за этот шаг для вычисления награды за эпизод
        reward = np.mean(reward_list) + self.usability_reward
        self.reward_data.append(reward)
        self.episode_reward += reward

        # Подготовляем данные для сохранения в буфере воспроизведения
        actions_next = []
        observations_next = []
        # Если эпизод не завершился, то можно найти новые действия и состояния
        if not terminated:
            for agent_id in range(n_agents):
                w = widjets[agent_id]
                # Получаем новое состояние среды для независимого агента
                cur_obs_agent = self.get_obs_agent(w.agent)
                obs_size = len(cur_obs_agent)
                obs_agent_next[agent_id] = cur_obs_agent
                # Собираем от разных агентов новые состояния
                for i in range(obs_size): observations_next.append(obs_agent_next[agent_id][i])
                # Конвертируем данные в тензор
                obs_agent_nextT = torch.FloatTensor([obs_agent_next[agent_id]]).to(self.device)
                # Получаем новые действия агентов для новых состояний из целевой сети исполнителя
                action_probabilitiesT = w.target_net(obs_agent_nextT)
                # Конвертируем данные в numpy
                action_probabilitiesT = action_probabilitiesT.to("cpu")
                action_probabilities = action_probabilitiesT.data.numpy()[0]
                # Находим новые возможные действия агента
                avail_actions_ind = w.available_actions()
                # Выбираем новые возможные действия
                action = self.select_actionFox(action_probabilities, avail_actions_ind, n_actions, noise_rate)
                # Собираем новые действия от разных агентов
                actions_next.append(action)
        elif terminated:
            # если эпизод на этом шаге завершился, то новых действий не будет
            actions_next = actions
            observations_next = observations

        # Сохраняем переход в буфере воспроизведения
        self.experience_buffer.append([observations, actions, observations_next, actions_next, reward, terminated])

        # Если буфер воспроизведения наполнен, начинаем обучать сеть
        ########################_начало if обучения_#######################
        if (self.global_step % self.app.steps_train == 0) and (self.global_step > self.app.start_steps):
            # Получаем минивыборку из буфера воспроизведения
            exp_obs, exp_acts, exp_next_obs, exp_next_acts, exp_rew, exp_termd = self.sample_from_expbuf(self.experience_buffer,
                                                                                                    batch_size)
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
            y_batch = np.zeros([batch_size])
            action_probabilitieQBT = torch.empty(1, batch_size, dtype=torch.float)

            for i in range(batch_size):
                # Вычисляем целевое значение y
                y_batch[i] = exp_rew[i] + (self.app.gamma * action_probabilitieQ_next[i]) * (1 - exp_termd[i])
                action_probabilitieQBT[0][i] = action_probabilitieQT[0][i]

            y_batchT = torch.FloatTensor([y_batch])

            # Обнуляем градиенты
            self.optimizerCritic.zero_grad()

            # Вычисляем функцию потерь критика
            loss_t_critic = self.objectiveCritic(action_probabilitieQBT, y_batchT)

            # Сохраняем данные для графиков
            self.Loss_History.append(loss_t_critic)
            loss_n_critic = loss_t_critic.data.numpy()
            self.total_loss.append(loss_n_critic)
            self.m_loss.append(np.mean(self.total_loss[-1000:]))

            # Выполняем обратное распространение ошибки для критика
            loss_t_critic.backward()

            # Выполняем оптимизацию нейронной сети критика
            self.optimizerCritic.step()
            ###################_Закончили обучать критика_#################

            ##############_Обучаем нейронные сети исполнителей_############
            act_full = np.zeros([batch_size, n_agents])
            for agent_id in range(n_agents):
                w = widjets[agent_id]
                # Разбираем совместное состояние на локальные состояния
                obs_local = np.zeros([batch_size, obs_size])
                for i in range(batch_size):
                    k = 0
                    for j in range(agent_id*obs_size,(agent_id+1)*obs_size):
                        obs_local[i][k] = exp_obs[i][j]
                        k += 1

                # Конвертируем данные в тензор
                obs_agentT1 = torch.FloatTensor([obs_local]).to(self.device)

                # Обнуляем градиенты
                w.optimizer.zero_grad()

                # Подаем в нейронные сети исполнителей локальные состояния
                action_probabilitiesT1 = w.actor_network(obs_agentT1)

                # Конвертируем данные в numpy
                action_probabilitiesT1 = action_probabilitiesT1.to("cpu")
                action_probabilities1 = action_probabilitiesT1.data.numpy()[0]

                # Вычисляем максимальные значения с учетом объема минивыборки
                for i in range(batch_size):
                    act_full[i][agent_id] = np.argmax(action_probabilities1[i])

            act_fullT = torch.FloatTensor([act_full]).to(self.device)

            # Конвертируем данные в тензор
            exp_obs = [x for x in exp_obs]
            obs_agentsT = torch.FloatTensor([exp_obs]).to(self.device)

            # Задаем значение функции потерь для нейронных сетей исполнителей
            # как отрицательный выход критика
            actor_lossT = -self.critic_network(obs_agentsT, act_fullT)

            # Усредняем значение по количеству элементов минивыборки
            actor_lossT = actor_lossT.mean()

            # Выполняем обратное распространение ошибки
            actor_lossT.backward()

            # Выполняем оптимизацию нейронных сетей исполнителей
            for agent_id in range(n_agents): widjets[agent_id].optimizer.step()

            # Собираем данные для графиков
            actor_lossT = actor_lossT.to("cpu")
            self.Loss_History_actor.append(actor_lossT)
            actor_lossN = actor_lossT.data.numpy()
            self.total_loss_actor.append(actor_lossN)
            self.m_loss_actor.append(np.mean(self.total_loss_actor[-1000:]))
            ##############_Закончили обучать исполнителей_#################

            # Рализуем механизм мягкой замены
            # Обновляем целевую сеть критика
            for target_param, param in zip(self.tgtCritic_network.parameters(), self.critic_network.parameters()):
                target_param.data.copy_((1 - self.app.TAU) * param.data + self.app.TAU * target_param.data)
            # Обновляем целевые сети акторов
            for agent_id in range(n_agents):
                for target_param, param in zip(widjets[agent_id].target_net.parameters(),
                                               widjets[agent_id].actor_network.parameters()):
                    target_param.data.copy_((1 - self.app.TAU) * param.data + self.app.TAU * target_param.data)

            ######################_конец if обучения_######################

        self.global_step += 1
        self.Reward_History.append(self.episode_reward)
        if self.is_done():
            self.app.stop_emulation_async('Adapt is stopped. End of episode!', 'Adapt', 0)

    def set_emulation(self, on=False):
        method = self.usability_reward_update if self.mode == 'IQL' else self.MADDPG_emulation
        time = 1. / 2. if self.mode=='IQL' else 1. / 30.
        if on:
            Clock.schedule_interval(method, time)
            return True
        else:
            Clock.unschedule(method)
            return False

    def change_emulation(self):
        self.emulation = self.set_emulation(True) if not(self.emulation) else self.set_emulation(False)

    def start_emulation(self):
        self.emulation = self.set_emulation(True)

    def stop_emulation(self):
        self.emulation = self.set_emulation(False)

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
            return action

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
        reward, done, rik = e.take_action(action, self)

        # Получаем новое состояние среды
        next_state = e.get_obs_agent(self)

        # Сохраняем переход в буфере воспроизведения для каждого агента
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
        return reward, rik

class Agent4:
    loss_data = [0]
    m_loss = [0]
    widget=None
    device=None

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
        reward = 0
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
        self.sm_layer = nn.Softmax(dim=1)

    #Вначале данные x обрабатываются полносвязной сетью с функцией ReLU
    #На выходе происходит обработка функцией Tanh()
    def forward(self, x):
        #Обработка полносвязными линейными слоями
        network_out = self.MADDPG_Actor(x)
        #Обработка функцией Tanh()
        tanh_layer_out = self.tanh_layer(network_out)
        sm_layer_out = self.sm_layer(network_out)
        #Выход нейронной сети
        return sm_layer_out

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
        self.sm_layer = nn.Softmax(dim=1)

    #Данные x последовательно обрабатываются полносвязной сетью с функцией ReLU
    def forward(self, state, action):
        #Объединяем данные состояний и действий для передачи в сеть
        x = torch.cat([state, action], dim=2)
        #Результаты обработки
        Q_value = self.network(x)
        sm_layer_out = self.sm_layer(Q_value)
        #Финальный выход нейронной сети
        return sm_layer_out # Q_value

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
        noise_rate = max(self.noise_rate_min, self.noise_rate_max - (self.noise_rate_max - self.noise_rate_min) * current_step / self.noise_decay_steps)
        return noise_rate

def get_optimizer_Adam(policy_net, lr):
    return optim.Adam(params=policy_net.parameters(), lr=lr)

def get_MSELoss_func():
    return nn.MSELoss()

def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")