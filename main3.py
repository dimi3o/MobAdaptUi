import random
import torch
import agent
from collections import deque
from dqnvianumpy.q_learning import train_main
from dqnvianumpy.q_learning import test_main
from colors import MyColors
from kivywidgets import Widgets, FlyScatter, FlyScatterV3, AsyncConsoleScatter
from kivy.app import App
from kivy.properties import NumericProperty
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.uix.spinner import Spinner
from kivy.uix.slider import Slider
from kivy.uix.checkbox import CheckBox
from kivy.uix.scatter import Scatter
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelHeader
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy_garden.graph import LinePlot

WhiteBackColor = True
__version__ = '0.0.3.3'

class MainApp(App):
    sm = ScreenManager()
    FlyScatters = []
    IdsPngs = [j for j in range(1, 41)]
    AdaptUiOnOff = False
    y_discount = 0.999
    total_reward = 0
    rewards_count = 0
    reward_data = [0. for i in range(40)]
    cumulative_reward_data = [0. for i in range(40)]
    loss_data = [0. for i in range(40)]
    m_loss_data = [0. for i in range(40)]
    target_ui_vect = [[0. for j in range(4)] for i in range(40)]
    current_ui_vect = [[0. for j in range(4)] for i in range(40)]
    sliders_reward = []
    strategy = None
    #DQN hyperparameters
    batch_size = 20 #256
    gamma = y_discount
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001
    eps_decay_steps = 500
    target_update = 50
    TAU = 0.005 # TAU is the update rate of the target network
    memory_size = 1000
    lr = 0.01
    steps_learning = 1

    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.title = 'MARL Mobile User Interface v.'+__version__
        self.text_color = MyColors.get_textcolor(WhiteBackColor)
        self.modes = ('Fly adapt', 'Size adapt', 'Rotate adapt', 'Fly+Size+Rotate adapt','DQN adapt', 'GAN adapt')
        self.cols_rows = ('1х1', '2х2', '3х3', '4х4', '5х5', '6х6', '8x5')
        self.objects = ('Apps', 'Foods', 'Widgets')
        self.kitchen = ('rus', 'eur', 'asia')
        self.episodes = ('200', '2000', '20000', '200000')
        self.modeargs = ('train', 'test')
        self.r_modeargs = ('map', 'weights', 'stats')

    def on_resize_my(self, oldsize, newsize):
        self.reward_graph.height = self.graph_layout.height = Window.height * 5 / 7

    def build(self):
        Window.bind(size=self.on_resize_my)

        # MAIN SCREEN
        self.root = BoxLayout(orientation='vertical', padding=10)  # ,size_hint_y=None)

        #HEAD PANEL
        self.colrowspinner = Spinner(text=self.cols_rows[6], values=self.cols_rows, background_color=(0.527, 0.154, 0.861, 1))
        self.colrowspinner.bind(text=self.colrowspinner_selected_value)
        self.objectspinner = Spinner(text=self.objects[1], values=self.objects, background_color=(0.027, 0.954, 0.061, 1))
        self.objectspinner.bind(text=self.colrowspinner_selected_value)
        lbl = Label(text='Size/Objs:', color=(0, 0, 1, 1))
        btn = Button(text='rebuild', size_hint_y=None, height='30dp', on_press=self.mainscreen_rebuild_btn_click)
        self.grid_flag = CheckBox(active=False, color=(0, 0, 1, 1), size_hint_x=None, width='30dp')
        self.headpanel = self.ihbl([lbl, self.colrowspinner, self.objectspinner, btn, self.grid_flag])
        self.root.add_widget(self.headpanel)
        self.headpanel.bind(size=self._update_rect_headpanel, pos=self._update_rect_headpanel)
        with self.headpanel.canvas.before:
            Color(0.827, 0.827, 0.827, 1.)
            self.rect_headpanel = Rectangle()

        #FOOT PANEL
        self.episodespinner = Spinner(text=self.episodes[1], values=self.episodes, size_hint_x=None, width='50dp', background_color=(0.225, 0.155, 0.564, 1))
        self.modespinner = Spinner(text="DQN adapt", values=self.modes, background_color=(0.127,0.854,0.561,1))
        self.adapt_btn = Button(text='ADAPT UI', size_hint_y=None, height='30dp', background_color=(1, 0, 0, 1), on_press=self.adapt_ui) #on_press=lambda null: self.show_popup('MARLMUI starting... '+self.modespinner.text, 'Info'))
        test_btn = Button(text='TEST UI', size_hint_y=None, height='30dp', size_hint_x=None, width='100dp', background_color=(0, 0, 1, 1), on_press=lambda null: self.adapt_ui(self, False))
        quit_btn = Button(text='QUIT', size_hint_y=None, height='30dp', background_color=(0.9, 0.9, 0.9, 1), on_press=lambda null: self.get_running_app().stop())
        sett_btn = Button(text='SETTINGS', size_hint_y=None, height='30dp', background_color=(0.2, 0.2, 0.2, 1), on_press=lambda null: self.to_screen('settings', 'left'))
        self.footpanel = self.ihbl([quit_btn, self.modespinner, self.episodespinner, self.adapt_btn, test_btn, sett_btn])
        self.root.add_widget(self.footpanel)
        self.footpanel.bind(size=self._update_rect_footpanel, pos=self._update_rect_footpanel)
        with self.footpanel.canvas.before:
            Color(0.827, 0.827, 0.827, 1.)
            self.rect_footpanel = Rectangle()

        # MAIN CONTENT
        self.mainscreen_widgets = BoxLayout(orientation='vertical', padding=0, spacing=0)
        self.mainscreen_rebuild_btn_click(self)
        self.root.add_widget(self.mainscreen_widgets)

        # SWAP: main content is top layer and center of screen
        swap = self.root.children[0]
        self.root.children[0] = self.root.children[1]
        self.root.children[1] = swap

        self.add_screen('mainscreen', self.root)

        # SETTINGS SCREEN
        self.root1 = BoxLayout(orientation='vertical', padding=10)
        self.root2 = BoxLayout(orientation='vertical', padding=10) #graph tab
        self.root3 = BoxLayout(orientation='vertical', padding=10) #reward func tab
        self.root4 = BoxLayout(orientation='vertical', padding=10) #dqn test

        # SETTINGS SCREEN, reward func tab
        self.targetUiVect = TextInput(password=False, multiline=True, readonly=True)
        cur_sl_label = Label(text='_._', color=(1, 0, 0, 1))
        temp_sliders = [cur_sl_label]
        for i in range(7):
            param = 'nx' if i==0 else 'ny' if i==1 else 'ns' if i==2 else 'nr' if i==3 else 'nt' if i==4 else 'penalty+'if i==5 else 'penalty-'
            sl_label = Label(text=f'R{str(i+1)} ({param}):', color=(0, 0, 0, 1), size_hint_x=None, width='80dp')
            value = 0.01 if i == 5 else .95 if i == 4 else .15 if i == 3 else 0.55 if i < 5 else -.95
            slider = Widgets.get_random_widget('Slider', -2., 2., value, 0.01)
            labelcallback = lambda instance, value: self.OnSliderRewardChangeValue(cur_sl_label, value)
            slider.bind(value=labelcallback)
            temp_sliders.append(self.ihbl([sl_label, slider], my_height=False))
            self.sliders_reward.append(slider)

        self.root3.add_widget(self.ihbl([self.ivbl(temp_sliders),self.ivbl([Label(text='Target UI State vector:',color=(1, 0, 0, 1),size_hint_y=None,height='30dp'),self.targetUiVect], my_width=False)], my_height=False))
        self.kitchenspinner = Spinner(text=self.kitchen[0], values=self.kitchen, background_color=(0.027, 0.954, 0.061, 1))
        self.kitchenspinner.bind(text=self.target_ui_selected_value)
        self.root3.add_widget(self.ihbl([Label(text='Kitchen:', color=(0, 0, 1, 1)),self.kitchenspinner]))

        # SETTINGS SCREEN, graph tab
        self.graph_widget_id = Spinner(text='1', values=[str(j) for j in range(1, 41)], background_color=(0.327, 0.634, 0.161, 1))
        self.root2.add_widget(self.ihbl([Label(text='REWARD/LOSS graph for widget id:', color=(0, 0, 0, 1)), self.graph_widget_id]))
        self.reward_graph = Widgets.get_graph_widget(.5, .5, 0, .1, 0, .1, 'Time, [sec]', WhiteBackColor)
        self.graph_layout = BoxLayout(orientation='horizontal', size_hint_y=None)
        self.reward_graph.height = self.graph_layout.height = Window.height*5/7
        self.graph_layout.add_widget(self.reward_graph)
        self.root2.add_widget(self.graph_layout)

        # SETTINGS SCREEN, dqn test
        self.console = TextInput(password=False, multiline=True, readonly=True)
        self.root4.add_widget(self.console)
        self.dqnmodespinner = Spinner(text=self.modeargs[0], values=self.modeargs, background_color=(0.027, 0.954, 0.061, 1))
        self.dqnr_modespinner = Spinner(text=self.r_modeargs[0], values=self.r_modeargs, background_color=(0.027, 0.125, 0.061, 1))
        self.test_dqn_btn = Button(text='TEST', size_hint_y=None, height='30dp', background_color=(1, 0, 0, 1), on_press=self.test_dqn)
        self.root4.add_widget(self.ihbl([Label(text='Mode:', color=(1, 0, 1, 1)), self.dqnmodespinner, self.dqnr_modespinner, self.test_dqn_btn]))

        tp = TabbedPanel(do_default_tab=False, background_color=(0,0,0,0))
        reward_th = TabbedPanelHeader(text='Graph')
        tp.add_widget(reward_th)
        reward_th.content = self.root2
        uivect_th = TabbedPanelHeader(text='Reward Func')
        tp.add_widget(uivect_th)
        uivect_th.content = self.root3
        dqnvect_th = TabbedPanelHeader(text='DQN test')
        tp.add_widget(dqnvect_th)
        dqnvect_th.content = self.root4
        self.root1.add_widget(tp)

        btn2 = Button(text='MAIN SCREEN', size_hint_y=None, height='30dp', background_color=(0.2, 0.2, 0.2, 1),
                      on_press=lambda null: self.to_screen('mainscreen', 'right'))
        self.root1.add_widget(btn2)
        self.target_ui_selected_value(self.kitchenspinner, self.kitchenspinner.text)
        self.add_screen('settings', self.root1)

        if WhiteBackColor:
            self.sm.bind(size=self._update_rect, pos=self._update_rect)
            with self.sm.canvas.before:
                Color(1, 1, 1, 1)
                self.rect = Rectangle(size=self.root.size, pos=self.sm.pos)

        return self.sm

    def OnSliderRewardChangeValue(self, label, value): label.text=f"{value:.{2}f}"

    # Need Asynchronous app
    # https://kivy.org/doc/stable/api-kivy.app.html#module-kivy.app
    def test_dqn(self, instance):
        #self.cclear()
        if self.dqnmodespinner.text == "train":
            train_main(self.dqnr_modespinner.text, self)
        else:
            test_main('model.pkl', self)

    def cwriteline(self, string = ''):
        self.console.text += '\n'+string

    def cwrite(self, string = ''):
        self.console.text += string

    def cclear(self):
            self.console.text = ''

    def add_screen(self, name, widget):
        scr = Screen(name=name)
        scr.add_widget(widget)
        self.sm.add_widget(scr)

    def to_screen(self, namescreen='mainscreen', direction='right'):
        self.sm.transition.direction = direction
        self.sm.current = namescreen
        #if namescreen == 'settings': print(self.current_ui_vect)

    def adapt_ui(self, instance, learning=True):
        m = self.modespinner.text
        if m == 'GAN adapt':
            self.show_popup('This adapt ui in the pipeline...', self.modespinner.text)
            return

        self.AdaptUiOnOff = not self.AdaptUiOnOff
        if m == 'DQN adapt' and self.AdaptUiOnOff == True:
            self.total_reward = 0
            self.rewards_count = 0
        for s in self.FlyScatters:
            s.change_emulation()
            s.mode = self.modespinner.text
            if m == 'DQN adapt' and s.emulation:
                s.agent.total_reward = 0
                s.agent.current_step = 0
                s.env.steps_left = int(self.episodespinner.text)
                s.env.done = False
                s.env.steps_learning = int((int(self.episodespinner.text) -self.batch_size ) * self.steps_learning) if learning else 0

            self.adapt_btn.background_color = (0.127,0.854,0.561,1) if s.emulation else (1, 0, 0, 1)
        if self.AdaptUiOnOff:
            Clock.schedule_interval(self._update_clock, 1 / 8.)
            self.total_reward = 0
            self.reset_reward_graph()
        else:
            Clock.unschedule(self._update_clock)
            print('- adapt ui stopped -')

    def stop_emulation_async(self,Text='Stop emulation!', Header='Adapt', par=0):
        if self.AdaptUiOnOff:
            self.AdaptUiOnOff = False
            Clock.unschedule(self._update_clock)
            self.adapt_btn.background_color = (1, 0, 0, 1)
            #self.show_popup(Text, Header)
            #print('- end of episode -')

    def _update_clock(self, dt):
        widget_id = int(self.graph_widget_id.text)-1
        #reward = self.reward_data[widget_id]
        reward = self.cumulative_reward_data[widget_id]
        #reward = self.m_loss_data[widget_id]
        loss = self.loss_data[widget_id]
        #reward = self.total_reward
        #reward = self.total_reward / self.rewards_count
        self.reward_points.append((self.reward_graph.xmax, reward))
        self.loss_points.append((self.reward_graph.xmax, loss))
        self.reward_plot.points = self.reward_points
        self.loss_plot.points = self.loss_points
        self.reward_graph.xmax += 1 / 8.
        self.expand_graph_axes(self.reward_graph, new_ymax=reward)
        self.expand_graph_axes(self.reward_graph, new_ymax=loss)

    def expand_graph_axes(self, graph, new_ymax=1.):
        if new_ymax==0: return
        new_ymax = float(new_ymax)
        if graph.ymax < new_ymax: graph.ymax = new_ymax*1.2
        elif graph.ymin > new_ymax: graph.ymin = new_ymax*0.8 if new_ymax>0 else new_ymax*1.2
        if abs(new_ymax) > graph.y_ticks_major * 20: graph.y_ticks_major *= 4
        if graph.xmax > graph.x_ticks_major * 20: graph.x_ticks_major *= 2

    def reset_reward_graph(self):
        try:
            self.reward_graph.remove_plot(self.reward_plot)
            self.reward_graph.remove_plot(self.loss_plot)
        except: pass
        self.reward_plot = LinePlot(line_width=2, color=[1, 0, 0, 1])
        self.loss_plot = LinePlot(line_width=2, color=[0, 0, 1, 1])
        self.reward_graph.add_plot(self.reward_plot)
        self.reward_graph.add_plot(self.loss_plot)
        self.reward_points = []
        self.loss_points = []
        self.reward_graph.x_ticks_major = .5; self.reward_graph.y_ticks_major = .1
        self.reward_graph.xmin = 0; self.reward_graph.xmax = .1; self.reward_graph.ymin = 0; self.reward_graph.ymax = .1

    def do_current_ui_vect(self, vect):
        self.current_ui_vect[vect[0]-1] = [vect[2], vect[3], vect[4], vect[5]]

    def target_ui_selected_value(self, spinner, text):
        self.target_ui_vect = Widgets.target_ui(text)
        print(self.target_ui_vect[1][1])
        self.targetUiVect.text = str(self.target_ui_vect).replace(' ','')

    def colrowspinner_selected_value(self, spinner, text):
        self.mainscreen_rebuild_btn_click(self)

    def mainscreen_rebuild_btn_click(self, instance):
        self.mainscreen_widgets.clear_widgets()
        self.FlyScatters.clear()
        TextSize = self.colrowspinner.text
        Objects = self.objectspinner.text
        rows = int(TextSize[0])
        cols = int(TextSize[2])
        random.shuffle(self.IdsPngs)

        #print('W =', Window.width, ',w =', Window.width // cols, ',H =', Window.height, ',h =', Window.height // (rows + 1))

        for i in range(rows):
            hor = BoxLayout(orientation='horizontal', padding=0, spacing=0)
            for j in range(cols):
                s = FlyScatterV3(do_rotation=True, do_scale=True, auto_bring_to_front=False, do_collide_after_children=False)
                s.app = self
                self.DQN_init2(s) #### DQN INIT
                hor.add_widget(s)
                s.id = ids = self.IdsPngs[i*cols+j]
                s.grid_rect = Widgets.get_random_widget('LineRectangle', 0, 0, Window.width // cols, Window.height // (rows + 1), f'S{i*cols+j}')
                if self.grid_flag.active: s.add_widget(s.grid_rect)
                w = Widgets.get_app_icon(ids) if Objects=='Apps' else Widgets.get_food_icon(ids) if Objects=='Foods' else Widgets.get_random_widget()
                w.width = f'{360 // cols}dp'#f'{Window.width//cols}dp'
                w.height = f'{800 // rows}dp'#f'{Window.height//(rows+3)}dp'
                s.add_widget(w)
                wi = Widgets.get_random_widget('Label')
                s.add_widget(wi)
                wi.text = str(ids)
                s.raw_width = w.width
                s.raw_height = w.height
                s.raw_rotate = s.rotation
                self.FlyScatters.append(s)

            self.mainscreen_widgets.add_widget(hor)

    def DQN_init(self, s):
        #### DQN INIT start
        s.app.strategy = agent.EpsilonGreedyStrategy(self.eps_start, self.eps_end, self.eps_decay, self.eps_decay_steps)
        steps_left = int(self.episodespinner.text)
        steps_learning = int((int(self.episodespinner.text) - self.batch_size) * self.steps_learning)
        s.env = agent.Environment(steps_left, steps_learning, self, s)
        s.set_vect_state()
        s.agent = agent.Agent(s.app.strategy, s.env.num_actions_available())
        s.memory = agent.ReplayMemoryPyTorch(self.memory_size)
        s.policy_net = agent.get_nn_module(s.env.num_state_available() - 2, s.agent.device)
        s.target_net = agent.get_nn_module(s.env.num_state_available() - 2, s.agent.device)
        s.target_net.load_state_dict(s.policy_net.state_dict())
        s.target_net.eval()
        s.optimizer = agent.get_optimizer_Adam(s.policy_net, self.lr)
        #### DQN INIT end

    def DQN_init2(self, s):
        #### DQN INIT start
        s.app.strategy = agent.EpsilonGreedyStrategy(self.eps_start, self.eps_end, self.eps_decay, self.eps_decay_steps)
        steps_left = int(self.episodespinner.text)
        steps_learning = int((int(self.episodespinner.text) - self.batch_size) * self.steps_learning)
        s.env = agent.Environment(steps_left, steps_learning, self, s)
        s.set_vect_state()
        memory = agent.ReplayMemoryPyTorch(self.memory_size)
        s.agent = agent.Agent2(s.app.strategy, memory, s.env.num_actions_available())
        s.experience_buffer = deque(maxlen=self.memory_size)
        s.policy_net = agent.get_nn_module2(s.env.num_state_available() - 2, s.agent.device)
        s.target_net = agent.get_nn_module2(s.env.num_state_available() - 2, s.agent.device)
        s.target_net.eval()
        s.optimizer = agent.get_optimizer_AdamW(s.policy_net, self.lr)
        #### DQN INIT end

    def ihbl(self, widjets, my_height=True):
        hor = BoxLayout(orientation='horizontal', padding=0, spacing=0)
        if my_height:
            hor.size_hint_y=None;  hor.height='30dp'
        for w in widjets: hor.add_widget(w)
        return hor

    def ivbl(self, widjets, my_width=True):
        vert = BoxLayout(orientation='vertical', padding=0, spacing=0)
        if my_width:
            vert.size_hint_x=None;  vert.width='400dp'
        for w in widjets: vert.add_widget(w)
        return vert

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def _update_rect_headpanel(self, instance, value):
        self.rect_headpanel.pos = (instance.pos[0]-5, instance.pos[1]-5)
        self.rect_headpanel.size = (instance.size[0]+10, instance.size[1]+10)

    def _update_rect_footpanel(self, instance, value):
        self.rect_footpanel.pos = (instance.pos[0]-5, instance.pos[1]-5)
        self.rect_footpanel.size = (instance.size[0]+10, instance.size[1]+10)

    def show_popup(self, text='', title='Popup Window'):
        popup = Popup(title=title, size_hint=(None, None),
                      size=(Window.width / 2, Window.height / 4))
        layout = BoxLayout(orientation='vertical', padding=10)
        layout.add_widget(Label(text=text))
        layout.add_widget(Button(text='OK', on_press=popup.dismiss))
        popup.content = layout
        popup.open()

if __name__ == "__main__":
    # Window.minimum_height = 800
    # Window.minimum_width = 300
    # Window.on_resize(300, 800)
    app = MainApp()
    app.run()
    # x = torch.randn(5)
    # y = torch.randn(5)
    # print(torch.cat([], 0))