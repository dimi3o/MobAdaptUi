import random
import math as m
from kivy.metrics import dp
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.slider import Slider
from kivy.uix.switch import Switch
from kivy.uix.spinner import Spinner
from kivy.uix.progressbar import ProgressBar
from kivy.uix.behaviors import TouchRippleBehavior
from kivy.uix.scatter import Scatter
from kivy.properties import ListProperty, BooleanProperty, NumericProperty, StringProperty
from kivy.core.window import Window
from kivy.clock import Clock
from kivy_garden.graph import Graph, LinePlot
from colors import allcolors

widgets = ['Image','TextInput','Label','Button','CheckBox','Slider','Switch','Spinner','ProgressBar','FlyLabel','FlatButton']

class Widgets(object):
    @staticmethod
    def get_random_widget(name='', *params):
        ret = random.choice(widgets) if (name == '') else name
        if ret == 'Image': return Image(source='data/icons/bug1.png', allow_stretch=True, keep_ratio=True)
        elif ret == 'TextInput': return TextInput(text='textinput')
        elif ret == 'Label': return Label(text='label', color=random.choice(allcolors))
        elif ret == 'Button': return Button(text='button', background_color=random.choice(allcolors))
        elif ret == 'CheckBox': return CheckBox(active=True)
        elif ret == 'Slider':
            if len(params)>0: return Slider(min=params[0], max=params[1], value=params[2], step=params[3])
            return Slider(min=1, max=10, value=1, step=1)
        elif ret == 'Switch': return Switch(active=True)
        elif ret == 'Spinner': return Spinner(text="Spinner", values=("Python", "Java", "C++", "C", "C#", "PHP"), background_color=(0.784,0.443,0.216,1))
        elif ret == 'ProgressBar': return ProgressBar(max=1000, value=750)
        elif ret == 'FlyLabel': return FlyLabel()
        elif ret == 'FlatButton': return FlatButton().build()
        elif ret == 'LineRectangle': return LineRectangle(line_width=2, line_rect=[params[0], params[1], params[2], params[3]], line_color=[1,0,0,1], label_text=params[4], label_pos=[params[2]//3, 0])
        return ''

    @staticmethod
    def get_app_icon(id=''):
        id = random.randint(1,41) if (id == '') else id
        return Image(source="data/icons/apps/a"+str(id)+".png", allow_stretch=True, keep_ratio=True)

    @staticmethod
    def get_food_icon(id=''):
        id = random.randint(1, 41) if (id == '') else id
        return Image(source="data/icons/foods/f" + str(id) + ".png", allow_stretch=True, keep_ratio=True)

    @staticmethod
    def get_widget(name):
        return Widgets.get_random_widget(name)

    @staticmethod
    def get_graph_widget(x_ticks_major=5, y_ticks_major=5, xmin=0, xmax=1, ymin=0, ymax=30, xlabel='Time',
                         WhiteBackColor=True):
        graph_theme = {
            'label_options': {
                'color': [0, 0, 0, 1],  # color of tick labels and titles
                'bold': False},
            'background_color': [1, 1, 1, 1],  # back ground color of canvas
            'tick_color': [0, 0, 0, 1],  # ticks and grid
            'border_color': [0, 0, 0, 1]}  # border drawn around each graph
        if WhiteBackColor:
            graph = Graph(**graph_theme)
        else:
            graph = Graph()
        if xlabel is not None: graph.xlabel = xlabel
        graph.x_ticks_minor = 0
        graph.x_ticks_major = x_ticks_major; graph.y_ticks_major = y_ticks_major
        graph.y_grid_label = True; graph.x_grid_label = True
        graph.padding = 5
        graph.x_grid = True; graph.y_grid = True
        graph.xmin = xmin; graph.xmax = xmax; graph.ymin = ymin; graph.ymax = ymax
        graph.tick_color = [0, 0, 0, 1]
        return graph

    @staticmethod
    def target_ui(name='rus'):
        if name == "rus":
            return [[0.15428171391601714, 0.384265564386792, 0.48458080858847374, 0.9942921716447548], [0.9839064932395777, 0.8228717856358929, 0.375000000000009, 1.0000000010834675e-06], [0.41115474145639513, 0.37307279010613154, 0.4643659582770843, 0.012643920249284846], [0.632182355308555, 0.9622334877635564, 0.3749999999999913, 1.0000000003063114e-06], [0.9237201693907782, 0.42936492509910207, 0.45301551188171785, 0.012573234346845319], [0.11528017478532467, 0.06549259657004836, 0.7807764064044134, 0.9489627411158725], [0.22549540628939596, 0.32143192616433974, 0.3947129238531708, 0.9881563111314371], [0.3996952912572406, 0.25656264966652564, 0.42525899206853685, 0.9994275026739331], [0.5750853383682896, 0.8406784463421042, 0.4220226439163873, 0.011182868885692698], [0.50289557574287, 0.23176996962471047, 0.685914760103082, 0.03226260476057896], [0.45553745772362075, 0.7918476665051165, 0.45931725266643414, 0.9863470772187485], [0.4321823553085493, 0.9686164664869591, 0.3750000000000001, 1.0000000006393783e-06], [0.3865220409993466, 0.8512774576402001, 0.42837396390111043, 0.9995938513851068], [0.8228940166622161, 0.8480822039395436, 0.43385023255235206, 0.004147062293050396], [0.9632168380671624, 0.8946802962741904, 0.375000000000009, 1.0000000010834675e-06], [0.1664921818582184, 0.4723578745498199, 0.44683826210671773, 0.014214763472146175], [0.4367800564579737, 0.34627604095504366, 0.3750000000000001, 1.0000000006393783e-06], [0.2103721157696212, 0.24421607509404078, 0.42235826232188806, 0.9973526210867735], [0.5738040997213841, 0.7812815488726593, 0.37371318265406156, 0.01978346378700513], [0.3649658935958331, 0.06860424313914089, 0.8300413830648842, 0.8427839030767071], [0.34942373461889314, 0.7803185941465333, 0.3750000000000001, 1.0000000003063114e-06], [0.3444201153096623, 0.4408950990469501, 0.5243700466212033, 0.014175638680621383], [0.5584458718674384, 0.34814593329789034, 0.7385345008407359, 0.015062261841328639], [0.9952860932414513, 0.9487756104544322, 0.40729100939898294, 0.002304607176664142], [0.6151167932338415, 0.7849119240303353, 0.4434991898106811, 0.005608919139018231], [0.8574697116303861, 0.9579781686146215, 0.375000000000009, 9.999999974752427e-07], [0.1958228003646909, 0.8186621411861102, 0.4144606864642274, 0.007892396057354423], [0.7139197269537862, 0.9766071140435497, 0.43681966909128944, 0.0027794956897916334], [0.20203955291721742, 0.9095185394259023, 0.4235686987598174, 0.0032753832424712304], [0.7130475774704695, 0.0636808138471058, 0.8513698125141683, 0.9522648735464764], [0.9714823914232635, 0.7597297848500889, 0.4205872327393562, 0.9866073091248099], [0.7741728087231406, 0.8042551371887702, 0.4247750067320786, 0.9972870664427651], [0.7700845548372962, 0.3088856495751062, 0.45758279444106253, 0.9999091146203816], [0.8673196690356411, 0.2582191427534169, 0.7203037099757409, 0.9952108678906131], [0.45084623371035176, 0.8192901469639452, 0.5656269492475656, 0.9611079854478524], [0.5448260334694625, 0.46436114733802275, 0.3749999999999913, 1.0000000003063114e-06], [0.5923954625652296, 0.8764142115615885, 0.548477942167908, 0.9962590111949294], [0.7491832496036155, 0.7826011104018942, 0.30854379201366594, 0.9939891306459717], [0.2367800564579743, 0.9691483813805759, 0.3750000000000001, 1.0000000000287557e-06], [0.2956990923305728, 0.8680813822372478, 0.45089040309907086, 0.006381545834408864]]
        elif name == 'eur':
            return [[0.784335900015443, 0.2882105423174487, 0.48458080858848773, 0.994292171644754], [0.00689577535542282, 0.7569146162222029, 0.3750000000000001, 9.999999996956888e-07], [0.25650174261305236, 0.844956419530566, 0.46436595827708155, 0.012643920249284235], [1.0620681891485235, 0.9611699353711374, 0.375000000000009, 1.0000000010834675e-06], [0.3251586261688696, 0.3607429427972962, 0.45301551188171646, 0.012573234346845485], [0.35799018284916806, 0.29925980367264543, 0.4383278551189697, 0.9884563896803794], [0.75883733338181, 0.15125031340633438, 0.39471292385316636, 0.9881563111314375], [0.4478802174542881, 0.8161565620224078, 0.5113127347541316, 0.9959885401918454], [1.0620681891485235, 0.9563827013285842, 0.375000000000009, 9.999999993626218e-07], [0.13788973729485798, 0.33358646500158295, 0.527392782920282, 0.9953091632962525], [1.0487634777428825, 0.9778168563615617, 0.4593172526664164, 0.9863470772187469], [1.0597693385738107, 0.9718082332434778, 0.375000000000009, 1.0000000010834675e-06], [0.9224961371728921, 0.5769314556104758, 0.42837396390109256, 0.999593851385108], [0.6378752577830555, 0.6482556918337918, 0.4338502325523698, 0.004147062293048676], [0.35632106271174463, 0.4446805736690115, 0.3750000000000001, 9.999999996956888e-07], [0.9162873051286416, 0.22649426582786758, 0.44683826210671995, 0.014214763472145675], [1.1011486489186384, 0.9452124885626266, 0.375000000000009, 9.999999993626218e-07], [0.13087156110295337, 0.4371756847694419, 0.49226576007210743, 0.9950164396346189], [0.35026580482624897, 0.12819832995549307, 0.49917621189638417, 0.011629325629720522], [0.21997305078657145, 0.9018303100790337, 0.5003174770490325, 0.9979502440826407], [1.052872786849673, 0.9643614247328396, 0.375000000000009, 9.999999993626218e-07], [0.9023969184062167, 0.1305862553529854, 0.5243700466212037, 0.01417563868062216], [0.4570390759664195, 0.44505696932783284, 0.4613906175142564, 0.9998868495403345], [0.707581023896251, 0.5679659072572105, 0.49255449129142487, 0.9893758967815839], [0.6560155497224391, 0.20129970118981938, 0.5396001470787524, 0.003609543412844418], [1.0574704879990982, 0.998403977924329, 0.375000000000009, 1.0000000010834675e-06], [0.8297959469212918, 0.650701236723012, 0.41446068646423034, 0.007892396057352535], [0.29558532569344736, 0.18914690010502586, 0.4368196690912849, 0.0027794956897909673], [0.25559784952780745, 0.48288023109584394, 0.5231877235881941, 0.996506915890221], [0.23448198225197447, 0.15851036090305387, 0.3750000000000001, 1.0000000000287557e-06], [1.025941289856437, 0.9305153256727886, 0.4205872327393562, 0.9866073091248082], [0.849994936391561, 0.5156969174672426, 0.4247750067320786, 0.9972870664427634], [0.25393324952225005, 0.7563741823751334, 0.5481583538777992, 0.9978648956398707], [0.43410892052911865, 0.9237432283000092, 0.4606430913839741, 0.007353735314162224], [1.0102984539220852, 0.9044167863034873, 0.5656269492475656, 0.9611079854478548], [1.0689647408726615, 0.9271273821796481, 0.375000000000009, 9.999999993626218e-07], [0.5248076354458163, 0.544740960096007, 0.5484779421679089, 0.9962590111949294], [1.102717204021364, 0.9404373919755696, 0.30854379201366594, 0.9939891306459717], [1.0873555454703625, 0.9531912119668822, 0.375000000000009, 9.999999993626218e-07], [0.5223779961015246, 0.3732059624248215, 0.4508904030990738, 0.006381545834407587]]
        elif name == 'asia':
            return [[0.9629890078758957, 0.19888636872894716, 0.375000000000009, 9.999999993626218e-07], [0.32413793103448624, 0.2228723404255323, 0.3750000000000001, 1.0000000003063114e-06], [0.8278951726249378, 0.20864220585396853, 0.375000000000009, 9.999999993626218e-07], [0.7793103448275818, 0.18351063829786937, 0.375000000000009, 9.999999993626218e-07], [0.3186464107262212, 0.7880125454048362, 0.6042301805654543, 0.026378069702707008], [0.7912027535345472, 0.13144519318138334, 0.5461593732577003, 0.9706468483123681], [0.9238643244503544, 0.24886850959123127, 0.375000000000009, 1.0000000010834675e-06], [0.82573043187321, 0.17037092672192714, 0.375000000000009, 1.0000000010834675e-06], [0.9210824048635988, 0.24763901042227188, 0.375000000000009, 9.999999993626218e-07], [0.7723476181879033, 0.22952575865556338, 0.3749999999999913, 1.0000000003063114e-06], [0.8125842975421729, 0.7369032127316935, 0.7000860423675548, 0.9903204406625825], [0.3448275862068971, 0.45585106382978746, 0.3750000000000001, 9.999999996956888e-07], [0.8830728551811824, 0.2059632641050563, 0.375000000000009, 1.0000000010834675e-06], [0.6997378555902666, 0.1894284325385718, 0.3749999999999913, 1.0000000003063114e-06], [0.3172413793103483, 0.31489361702127683, 0.3750000000000001, 1.0000000000287557e-06], [0.18535286665680092, 0.8436033031708874, 0.37499999999999994, 9.999999995291553e-07], [0.9379310344827562, 0.16968085106382672, 0.375000000000009, 1.0000000010834675e-06], [0.32235221407846426, 0.8609343140972079, 0.8589263041685538, 0.9941336039059276], [0.7846978476770069, 0.2236492849905651, 0.3749999999999735, 1.0000000010834675e-06], [0.851075586785372, 0.2217926734990963, 0.375000000000009, 1.0000000010834675e-06], [0.7701149425287387, 0.167553191489362, 0.3749999999999913, 1.0000000003063114e-06], [0.22670401139170565, 0.9228490644691221, 0.3750000000000001, 1.0000000006393783e-06], [0.05262806456903641, 0.3587030526451063, 0.3750000000000001, 1.0000000000287557e-06], [0.8431576989794917, 0.2586117176712012, 0.375000000000009, 9.999999993626218e-07], [0.9353281667402411, 0.20524319090293386, 0.375000000000009, 1.0000000010834675e-06], [0.8758620689655153, 0.18031914893616632, 0.375000000000009, 9.999999993626218e-07], [0.22669651382086145, 0.28350083030542117, 0.3750000000000001, 1.0000000003063114e-06], [0.4871999557271248, 0.27948538881185286, 0.3749999999999913, 9.999999993626218e-07], [0.3496481900188667, 0.3647259152720788, 0.3750000000000001, 9.999999996956888e-07], [0.7908679278707191, 0.222193051491826, 0.375000000000009, 9.999999993626218e-07], [0.8316963405948207, 0.5066215918122401, 0.8146031174657705, 0.005463547259067059], [0.917034032437196, 0.2512758507533208, 0.375000000000009, 9.999999993626218e-07], [0.8690352860501411, 0.2314079032398953, 0.375000000000009, 1.0000000010834675e-06], [0.19217936240249603, 0.3853446753813027, 0.3750000000000002, 1.0000000000287557e-06], [0.8081765498201905, 0.6132323482268716, 0.7901890615780218, 0.011771335648848635], [0.880459770114938, 0.2005319148936141, 0.375000000000009, 9.999999993626218e-07], [0.5086241226577342, 0.390383268817172, 0.3749999999999913, 1.0000000003063114e-06], [0.26344364246642554, 0.7016047417545491, 0.6403196062190377, 0.004763694987573952], [0.5333333333333357, 0.32765957446808697, 0.3749999999999913, 9.999999993626218e-07], [0.8153408209593663, 0.4122971445765512, 0.7175237653007455, 0.9702310551236164]]

class FlyScatterV3(Scatter):#(TouchRippleBehavior, Scatter):
    velocity = ListProperty([2, 1])
    emulation = BooleanProperty(False)
    mode = 'Fly adapt'
    app = None
    experience_buffer = None # Agent2
    agent = None
    policy_net = None
    target_net = None
    optimizer = None
    env = None
    id = 1
    grid_rect = None
    raw_height = 0
    raw_width = 0
    raw_rotate = 0
    reduceW = -1
    reduceH = -1
    deltaposxy = 1
    doublesize = BooleanProperty(False)
    taps = 0
    steps_learning = 0
    vect_state = []

    def __init__(self, **kwargs):
        super(FlyScatterV3, self).__init__(**kwargs)
        self.velocity[0] *= random.choice([-2, 2])
        self.velocity[1] *= random.choice([-2, 2])
        self.text = 'flyscatter'
        self.color = random.choice(allcolors)
        self.set_vect_state()

    def tap_event(self):
        self.taps += 1
        self.children[0].text = f'taps: {self.taps}'
        self.set_vect_state()
        # self.env.get_rewards()
        # rewards = self.env.last_reward[int(self.id)-1][:-1]
        # print(sum(rewards)/len(rewards), rewards)
        # print(self.vect_state[2:])

    def toFixed(self,numObj, digits=0): return f"{numObj:.{digits}f}"

    def DQN_adapt(self, *args):
        r = self.MARL_core()
        self.app.rewards_count += 1
        self.app.reward_data[int(self.id) - 1] = r  # self.agent.reward_data[-1]
        self.app.cumulative_reward_data[int(self.id) - 1] += r  # self.agent.reward_data[-1]
        self.app.loss_data[int(self.id) - 1] = self.agent.loss_data[-1]
        self.app.m_loss_data[int(self.id) - 1] = self.agent.m_loss[-1]
        if self.env.is_done():
            self.emulation = self.set_emulation(False)
            self.app.stop_emulation_async('DQN adapt is stopped. End of episode!', 'Adapt',
                                          self.agent.total_reward)
            # print(self.id, self.taps, self.nx, self.ny, self.ns, self.nr)
        # self.children[0].text = f'{self.toFixed(sum(self.vect_state[2:]) / len(self.vect_state[2:]), 2)}'

        v = self.vect_state
        self.app.current_ui_vect[v[0] - 1] = v[2:]

    def MARL_core(self):
        r = self.agent.step(self.env)

        # target_net_state_dict = self.target_net.state_dict()
        # policy_net_state_dict = self.policy_net.state_dict()
        # for key in policy_net_state_dict:
        #     target_net_state_dict[key] = policy_net_state_dict[key] * self.app.TAU + target_net_state_dict[key] * (1 - self.app.TAU)
        # self.target_net.load_state_dict(target_net_state_dict)
        

        # Синхронизируем веса основной и целевой нейронной сети каждые target_update шагов
        # if self.env.steps_left % self.app.target_update == 0:
        #     self.target_net.load_state_dict(self.policy_net.state_dict())
        #     # print('--- target_net update ---')

        if self.env.steps_left % self.app.target_update == 0:
            self.target_net.load_state_dict(self.policy_net)

        return r

    def set_vect_state(self):
        nx = (self.x + self.size[0]/self.scale) / Window.width
        ny = (self.y + self.size[1]/self.scale) / Window.height
        ns = min(1, max(0, (self.scale - 0.4) / (2 - 0.4)))
        nr = ((self.rotation-180)/180 + 1) / 2
        self.vect_state = [int(self.id), int(self.taps), nx, ny, ns, nr]
        return self.vect_state

    def update_vect_state_from(self, v):
        self.scale = v[2] * 1.6 + 0.4
        self.x = v[0] * Window.width - self.size[0]/self.scale
        self.y = v[1] * Window.height - self.size[1]/self.scale
        self.rotation = (v[3] * 2 - 1) * 180 + 180

    # norm value: nx = (x – мин(х)) / (макс(х) – мин(х))
    def norm_min_max(self, value, minv, maxv): return (value - minv) / (maxv - minv)

    # 0 - left, 1 - right, 2 - up, 3 - down, 4 - more, 5 - less
    def change_pos_size(self, to=0, deltapos=1, deltascale=0.01):
        r = self.app.sliders_reward[5].value # reward for action
        if to==0 and self.x>0: self.x -= deltapos
        elif to==1 and self.x+((self.width//2)*self.scale)<Window.width: self.x += deltapos
        elif to==2 and self.y+((self.height*1.5)*self.scale)<Window.height: self.y += deltapos
        elif to==3 and self.y>0: self.y -= deltapos
        elif to==4 and self.scale<2.: self.scale += deltascale
        elif to==5 and self.scale>0.4: self.scale -= deltascale
        elif to==6: self.rotation -= 1 # and (self.rotation>265 or self.rotation==0.)
        elif to==7: self.rotation += 1 # and self.rotation<110
        else: r = self.app.sliders_reward[6].value  # penalty for inaction
        # if to==0: self.x -= deltapos
        # elif to==1: self.x += deltapos
        # elif to==2: self.y += deltapos
        # elif to==3: self.y -= deltapos
        # elif to==4: self.scale += deltascale
        # elif to==5: self.scale -= deltascale
        # elif to==6: self.rotation -= 1 # and (self.rotation>265 or self.rotation==0.)
        # elif to==7: self.rotation += 1 # and self.rotation<110
        # else: r = self.app.sliders_reward[6].value # penalty for inaction
        self.set_vect_state()
        return r

    def available_actions(self):
        a = []
        if self.x>0: a.append(0)
        if self.x+((self.width//2)*self.scale)<Window.width: a.append(1)
        if self.y+((self.height*1.5)*self.scale)<Window.height: a.append(2)
        if self.y>0: a.append(3)
        if self.scale<2.: a.append(4)
        if self.scale>0.4: a.append(5)
        if self.rotation==0.: a.append(6); a.append(7)
        else:
            if self.rotation<150 or self.rotation<230: a.append(7)
            if self.rotation>230 or self.rotation>150: a.append(6)
        return a

    def change_emulation(self):
        self.emulation = self.set_emulation(True) if not(self.emulation) else self.set_emulation(False)

    def start_emulation(self):
        self.emulation = self.set_emulation(True)

    def stop_emulation(self):
        self.emulation = self.set_emulation(False)

    def set_emulation(self, on=False):
        method = self.DQN_adapt if self.mode == 'DQN' else self.simple_adapt
        if on:
            Clock.schedule_interval(method, 1. / 60.)
            return True
        else:
            Clock.unschedule(method)
            return False

    def on_touch_up(self, touch):
        if touch.grab_current == self: self.tap_event()
        #print(self.id, self.taps, self.nx, self.ny, self.ns, self.nr)
        # self.app.do_current_ui_vect([self.id, self.taps, self.nx, self.ny, self.ns, self.nr])
        super(FlyScatterV3, self).on_touch_up(touch)

    def simple_adapt(self, *args):
        if self.mode ==  'Rotate' or self.mode == 'Fly+Size+Rotate': self.rotation += random.choice([-1, 1])
        if self.mode == 'Fly' or self.mode == 'Fly+Size+Rotate':
            self.x += self.deltaposxy*self.velocity[0]
            self.y += self.deltaposxy*self.velocity[1]
            if self.x < 0 or (self.x + 2*self.width//3) > Window.width:
                self.velocity[0] *= -1
            if self.y < 0 or (self.y + 2*self.height//3) > Window.height:
                self.velocity[1] *= -1
        if self.mode == 'Size' or self.mode == 'Fly+Size+Rotate':
            w = self.children[1].width
            h = self.children[1].height
            if w < self.raw_width // 3: self.reduceW = 1
            elif w > self.raw_width: self.reduceW = -1
            if h < self.raw_height // 3: self.reduceH = 1
            elif h > self.raw_height: self.reduceH = -1
            self.children[1].width = w + self.reduceW
            self.children[1].height = h + self.reduceH

    # def on_touch_down(self, touch):
    #     if touch.grab_current == self: self.tap_event()
    #     super(FlyScatterV3, self).on_touch_down(touch)

    # def on_touch_move(self, touch):
        # if not self.doublesize:
        #     self.children[0].width *= 2
        #     self.children[0].height *= 2
        #     self.doublesize = True
        # else:
        #     self.children[0].width //= 2
        #     self.children[0].height //= 2
        #     self.doublesize = False

class AsyncConsoleScatter(Scatter):
    console = None

    def __init__(self, **kwargs):
        super(AsyncConsoleScatter, self).__init__(**kwargs)

    def start_emulation(self, console_widget):
        self.console = console_widget
        Clock.schedule_interval(self.update_console, 1. / 2.)

    def update_console(self, *args):
        self.console.text += '.'

KV = """
<FlatButton>:
    ripple_color: 0, 0, 0, 0
    background_color: 0, 0, 0, 0
    color: root.primary_color

    canvas.before:
        Color:
            rgba: root.primary_color
        Line:
            width: 1
            rectangle: (self.x, self.y, self.width, self.height)

Screen:
    canvas:
        Color:
            rgba: 0.9764705882352941, 0.9764705882352941, 0.9764705882352941, 1
        # Rectangle:
        #     pos: (self.x, self.y)
        #     size: self.size
"""
class FlatButton(TouchRippleBehavior, Button):
    primary_color = [0.12941176470588237, 0.5882352941176471, 0.9529411764705882, 1]

    def build(self):
        screen = Builder.load_string(KV)
        btn = FlatButton(text="flatbutton", ripple_color=(0, 0, 0, 0))
        screen.add_widget(btn)
        return screen

    def on_touch_down(self, touch):
        collide_point = self.collide_point(touch.x, touch.y)
        if collide_point:
            touch.grab(self)
            self.ripple_show(touch)
            return True
        return False

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            self.ripple_fade()
            return True
        return False


class FlyLabel(TouchRippleBehavior, Label):
    velocity = ListProperty([2, 1])
    emulation = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(FlyLabel, self).__init__(**kwargs)
        self.velocity[0] *= random.choice([-2, 2])
        self.velocity[1] *= random.choice([-2, 2])
        self.text = 'flylabel'
        self.color = random.choice(allcolors)

    def update_pos(self, *args):
        parent = self.parent
        parent.x += self.velocity[0]
        parent.y += self.velocity[1]
        if parent.x < 0 or (parent.x + 2*parent.width//2) > Window.width:
            self.velocity[0] *= -1
        if parent.y < 0 or (parent.y + 2*parent.height//2) > Window.height:
            self.velocity[1] *= -1
        parent.pos = [parent.x, parent.y]

    def on_touch_down(self, touch):
        if self.emulation:
            self.emulation = False
            Clock.unschedule(self.update_pos)
            return
        self.emulation = True
        Clock.schedule_interval(self.update_pos, 1. / 60.)

class FlyScatter(TouchRippleBehavior, Scatter):
    velocity = ListProperty([2, 1])
    emulation = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(FlyScatter, self).__init__(**kwargs)
        self.velocity[0] *= random.choice([-2, 2])
        self.velocity[1] *= random.choice([-2, 2])
        self.text = 'flylabel'
        self.color = random.choice(allcolors)

    def update_pos(self, *args):
        parent = self
        parent.x += self.velocity[0]
        parent.y += self.velocity[1]
        if parent.x < 0 or (parent.x + 2*parent.width//3) > Window.width:
            self.velocity[0] *= -1
        if parent.y < 0 or (parent.y + 2*parent.height//3) > Window.height:
            self.velocity[1] *= -1
        parent.pos = [parent.x, parent.y]

    def on_touch_down(self, touch):
        if not(self.emulation):
            Clock.schedule_interval(self.update_pos, 1. / 60.)
            self.emulation = True
        else:
            Clock.unschedule(self.update_pos)
            self.emulation = False

# Builder.load_string('''
# <LineRectangle>:
#     canvas:
#         Color:
#             rgba: .1, .1, 1, .9
#         Line:
#             width: 2.
#             rectangle: (self.x, self.y, self.width, self.height)
#     Label:
#         center: root.center
#         text: 'Rectangle'
# ''')
#
# class LineRectangle(Widget):
#     pass

Builder.load_string('''
<LineRectangle>:
    canvas:
        Color:
            rgba: root.line_color
        Line:
            width: root.line_width
            rectangle: root.line_rect
    Label:
        text: root.label_text
        pos: root.label_pos
        color: root.line_color
''')

class LineRectangle(Widget):
    line_color = ListProperty([])
    line_rect = ListProperty([])
    line_width = NumericProperty()
    label_text = StringProperty('')
    label_pos = ListProperty([])

    def __init__(self, **kwargs):
        self.line_color = kwargs.pop('line_color', [.1, .1, 1, .9])
        self.line_rect = kwargs.pop('line_rect', [0, 0, 50, 50])
        self.line_width = kwargs.pop('line_width', 1)
        self.label_text = kwargs.pop('label_text', 'Rectangle')
        self.label_pos = kwargs.pop('label_pos', [0, 0])
        super(LineRectangle, self).__init__()
#
# self.bbox1 = LineRectangle(line_wdth=2, line_rect=[100, 100, 100, 100], line_color=[1,0,0,1], label_text='bbox1', label_pos=[100, 100])

