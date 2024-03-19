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
            return [[0.15629341010245915,0.42270765485719397,0.48458080858847374,0.9942921716447548],[0.9551679227242683,0.7983438232995185,0.4775356596149375,0.9743277865828374],[0.4284879973332747,0.3851245353374863,0.5983879670937879,0.9401533891000169],[0.5983124815110991,0.9416773796040381,0.43885388519551627,0.986723624267056],[0.93090015448134,0.42282327947809695,0.5463026937816083,0.984843557508156],[0.02230897279019903,0.03931481186962952,0.9563019943551073,0.8673392482010425],[0.17944274892520048,0.3516935287787166,0.3947129238531709,0.9881563111314373],[0.44083466250226255,0.2777535559180334,0.4252589920685386,0.999427502673934],[0.6014783953887398,0.859653445304406,0.5574879487012057,0.9867456169134964],[0.5287099564976188,0.24620823504881711,0.7542352523608145,0.9830156462111556],[0.4220378194521192,0.7982695505703616,0.45931725266643453,0.9863470772187497],[0.3889325221271069,0.966987248712524,0.42119789545833664,0.9781969247490274],[0.5403739889408653,0.9054716846792089,0.42837396390110755,0.9995938513851058],[0.852939021332741,0.833296777920779,0.5026361066913204,0.9735870649115106],[0.9966441330953164,0.8969872019892614,0.4532260618489932,0.9875646465718091],[0.18466688169491943,0.5332909915892048,0.4106660333880717,0.9839906966332267],[0.37807838372113717,0.3507374211504151,0.45337227882289893,0.9658085738542708],[0.20007473210200968,0.27786575138273856,0.4223582623218878,0.9973526210867735],[0.5949539597231468,0.8199940782572748,0.4212469798963058,0.993579626144463],[0.316261842990045,0.03535423221322842,0.9996220180554333,0.7695915893534732],[0.21007564902183185,0.7997692955722051,0.3360958763620987,0.9758321374129195],[0.3128642711214104,0.4529570015509015,0.5308943288191174,0.9872402527187404],[0.6102382481735804,0.37676466941323955,0.8468422081933582,0.9898094478332853],[1.0530954965636041,0.9843254452791503,0.3816386541323873,0.9949551131404759],[0.7718286140749266,0.864556446632297,0.48218456529970233,0.9950487449792635],[0.8501499350883909,0.9787238965161972,0.41293465471658036,0.9981528377580976],[0.19810540007816485,0.8312503312298023,0.44350477646006503,0.9767601290862941],[0.5897509287496021,0.9532535890126799,0.5787874597515793,0.9802078146109737],[0.3267622632568359,0.8756248623524918,0.48710882659536536,0.9869842164019695],[0.7452321750105744,0.08000665277706982,0.8701447154535324,0.9216531196667562],[0.9659965787417555,0.7577916415607153,0.42058723273936816,0.9866073091248116],[0.7730395719636378,0.803851139109027,0.4247750067320881,0.9972870664427641],[0.7677457666152541,0.32589265758040825,0.45758279444107475,0.9999091146203833],[0.788157225210723,0.2265681354142326,1,0.9851421980659382],[0.4321213388515096,0.8128244485504509,0.5656269492475661,0.9611079854478517],[0.4927579748938714,0.4646594766938459,0.5816969944101884,0.9590295570637011],[0.6779049690855491,0.9097989382378731,0.5484779421679112,0.9962590111949314],[0.7448144956153572,0.7967588339368674,0.3085437920136642,0.9939891306459724],[0.17315654747073395,0.9695600464764056,0.45659577648733424,0.9880286305585387],[0.1924842857489375,0.9183200519117605,0.43774167085355564,0.9931591328593122]]
        elif name == 'eur':
            return [[0.8956394697680178,0.2937945061147101,0.502640512411792,0.9854671319735271],[0.1734706524367002,0.862999829099409,0.4187381260919224,0.9757274958988629],[0.3355367056949891,0.8196926271660282,0.5172658737890915,0.9929697498357263],[1.056290534988945,0.9558984052746282,0.38974082276537597,0.9888531332545755],[0.331813635339053,0.4216662806344884,0.43776784336302255,0.9965553102418672],[0.34189174177526443,0.3428459577137653,0.45082785511897433,0.944011945235935],[0.7208693309573383,0.15652191468411003,0.537997519638349,0.9985790800303556],[0.5153247941774753,0.8866140808066181,0.4550627347541309,0.9737663179696224],[1.0697233487822435,0.9689789163341795,0.39232107058692406,0.9923688815264003],[0.14530247555223283,0.3825849658604995,0.5169608102233006,0.9788739771172781],[1.019175878192162,0.9628716613328062,0.49681725266641863,0.9863470772187447],[1.0758689825325882,0.9522005247939757,0.3958919960707364,0.9815929523609936],[0.9679839215373195,0.6442968184375956,0.4033739639011057,0.971816073607328],[0.5894094500197042,0.6711773339218776,0.530335614845876,0.9748951539276678],[0.32163123971598184,0.480117555395701,0.5084792798112733,0.9958555959249673],[0.9062854763615298,0.20231768758565696,0.48808723948152793,0.9774514165033394],[1.1022639724894814,0.9317508472219231,0.3436412282479652,0.9983874377479405],[0.13579354967049,0.46363939623648137,0.525619020103246,0.9742598759027253],[0.3867335082476727,0.13835380514642623,0.5378492590549171,0.9849812078084028],[0.1805480979547149,0.9320998035319943,0.4849396330913689,0.9806871506302637],[1.0645251148645762,0.959876643468486,0.37500000000002065,0.9944434444444457],[0.8941528578184962,0.11123297510423087,0.5989155631953699,0.9834013629756245],[0.47729577135220574,0.48396239437009503,0.5274689345021786,0.9966682488408376],[0.7460589653264104,0.6133863003325397,0.5244811412706339,0.994996680114468],[0.7082714543892961,0.2539271042223325,0.5001030972941615,0.990958063051526],[1.085220753402504,0.9764800517468892,0.3990426103855939,0.9993487969585513],[0.847835253151532,0.7125982643883483,0.4627713413269816,0.9771517076156812],[0.2336704264021986,0.19977253931621683,0.495724706203609,0.9956534779839953],[0.20701147842173032,0.5474843562968993,0.48252241234421794,0.9817885840721677],[0.19074051066927475,0.12832701538629987,0.5209936228571143,0.9897715053255303],[1.036556536198485,0.9289760220413407,0.4554224904401261,0.9933571285561309],[0.9099188722530069,0.5627530708342968,0.49919629643566393,0.9970363048404844],[0.22561532720289854,0.7955277724960497,0.41074974740585696,0.9789038826757351],[0.38633320238450664,0.9290839246210875,0.4245500389008268,0.996879026765871],[1.0223168490780516,0.895774134928977,0.5468769492475521,0.9833364589965915],[1.0733294461951721,0.9368184134594596,0.41688873572072255,0.997572638329618],[0.5569785249460167,0.5870811272149687,0.5802524031093413,0.9878769068324421],[1.112812438368552,0.9769299006852384,0.30646971588651245,0.9833208149911337],[1.0864511953335099,0.920435265913993,0.4737489205518875,0.9868519963142799],[0.5040749892112603,0.3954445952943712,0.5081683043406492,0.9869811924932138]]
        elif name == 'asia':
            return[[0.8888862813863778,0.15269421296699776,0.5125932435159464,0.9868067693568788],[0.30158237228636176,0.21590533699288564,0.3920929962564511,0.982197145446398],[0.8183863769833517,0.14288520378255842,0.4084024978249332,0.9976533359716838],[0.7931030600910343,0.14255305279167269,0.3749999999999933,0.9999990000000006],[0.40007713755748653,0.8107717161079173,0.5349704040123509,0.9479513128576931],[0.735845471152767,0.1584907733574239,0.5461593732577161,0.9706468483123697],[0.8757240904267494,0.1760431874308792,0.3885969124553464,0.9940543910271351],[0.855412100107758,0.20990580468509165,0.34717019873792954,0.9914558200850112],[0.8540954563903475,0.19161642323556238,0.3825927687787133,0.994549585052069],[0.8945297704051732,0.16894216131520404,0.4066289337241748,0.995461230217171],[0.7109352301180977,0.7096401599087033,1,0.9634687570912629],[0.36794947291084695,0.44260766594977924,0.4112959339904618,0.9964857090748441],[0.8227818147999959,0.19744531337001825,0.33868767030704827,0.9858090639745374],[0.913915672435171,0.1739938305721302,0.36585338179651805,0.9973563505252141],[0.3524801765660715,0.2840121520910154,0.404562151883731,0.993008842190622],[0.16125729312895232,0.8379839547900515,0.5404025637091928,0.9950433030069974],[0.897254288037726,0.16883654694648936,0.4010925569161031,0.9942814733094082],[0.3229384086242926,0.8669547763807929,0.8589263041685535,0.9941336039059276],[0.8514440282137556,0.1399154062215413,0.3652847012309647,0.9960850336533938],[0.7771464799073219,0.15223488995723344,0.4368822922092697,0.9964587172030652],[0.8572356201797001,0.19391628224254231,0.34265805136705046,0.9903473687564132],[0.18085086360378663,0.9202816718819877,0.5229140810364795,0.9986436099926995],[0.14820474835673295,0.4487440573054947,0.4943595557832491,0.995032986893032],[0.867655056586805,0.17533997377358224,0.3225491245299484,0.9958292400202001],[0.8736672076096086,0.18288156622889634,0.45646794577968586,0.9958318226648604],[0.9414092381791973,0.19253776856297067,0.39633400151118053,0.9941910700496506],[0.18568097463260838,0.25172065511963904,0.45886521714209055,0.993517006401982],[0.46070395102568834,0.2783043392723975,0.4470727398372869,0.9874600078378108],[0.2934205933447613,0.33713990423183765,0.5046181637936028,0.9909069053243513],[0.8459879907889742,0.2105322062856511,0.2946398636715396,0.9911384429500256],[0.71277998691778,0.5116209442184415,0.938887815906852,0.9644329014318787],[0.8156436979742873,0.17016351607682423,0.5705602955440975,0.9942771914400548],[0.8857453330707344,0.16600821475671376,0.3804742309046725,0.9946735572170688],[0.14612790816012539,0.35856530262385317,0.44632640468146445,0.994746838308531],[0.6510442479691141,0.5712755409515068,1,0.9961308977604741],[0.8980452297102082,0.13844459387560476,0.4262630632891769,0.9904932810806789],[0.5100294615947746,0.37900030087398934,0.5123475457347365,0.9936356508990281],[0.24863033837447918,0.7221701786855785,0.6772150597610804,0.988605808212526],[0.543822851317542,0.31729396037427204,0.4683972019104477,0.9948012352350819],[0.6567021925321455,0.37401546480500875,1,0.9999149801821571]]

class FlyScatterV3(Scatter):
    velocity = ListProperty([2, 1])
    emulation = BooleanProperty(False)
    mode = 'Fly adapt'
    app = None
    experience_buffer = None # Agent2
    agent = None
    policy_net = None
    target_net = None
    actor_network = None
    optimizer = None # Оптимизатор НС
    objective = None # Функция потерь
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

    def toFixed(self,numObj, digits=0): return f"{numObj:.{digits}f}"

    def IQL_adapt(self, *args):
        r, rik = self.MARL_core()
        self.app.rewards_count += 1
        self.app.reward_data[int(self.id) - 1] = r
        self.app.reward_ik_data[int(self.id) - 1] = rik
        self.app.cumulative_reward_data[int(self.id) - 1] += r
        self.app.loss_data[int(self.id) - 1] = self.agent.loss_data[-1]
        self.app.m_loss_data[int(self.id) - 1] = self.agent.m_loss[-1]
        if self.env.is_done():
            self.emulation = self.set_emulation(False)
            self.app.stop_emulation_async('IQL adapt is stopped. End of episode!', 'Adapt',
                                          self.agent.total_reward)

        v = self.vect_state
        self.app.current_ui_vect[v[0] - 1] = v[2:]

    def MARL_core(self):
        r, rik = self.agent.step(self.env)

        if self.env.steps_left % self.app.target_update == 0:
            self.target_net.load_state_dict(self.policy_net)

        return r, rik

    def set_vect_state(self):
        nx = (self.x + self.size[0]/self.scale) / Window.width
        ny = (self.y + self.size[1]/self.scale) / Window.height
        ns = min(1, max(0, (self.scale - 0.4) / (2 - 0.4)))
        nr = abs((self.rotation - 180) / 360) * 2
        nt = min(1, self.taps / 10)
        self.vect_state = [int(self.id), nt, nx, ny, ns, nr]
        return self.vect_state

    def update_vect_state_from(self, v):
        self.scale = v[2] * 1.6 + 0.4
        self.x = v[0] * Window.width - self.size[0]/self.scale
        self.y = v[1] * Window.height - self.size[1]/self.scale
        self.rotation = (v[3] / 2) * 360 + 180

    # norm value: nx = (x – мин(х)) / (макс(х) – мин(х))
    def norm_min_max(self, value, minv, maxv): return (value - minv) / (maxv - minv)

    # 0 - left, 1 - right, 2 - up, 3 - down, 4 - more, 5 - less, 6 - rotate-, 7 - rotate+
    def change_pos_size(self, to=0, deltapos=1, deltascale=0.01):
        r = self.app.sliders_reward[5].value # reward for action
        if to==0 and self.x>0: self.x -= deltapos
        elif to==1 and self.x+((self.width//2)*self.scale)<Window.width: self.x += deltapos
        elif to==2 and self.y+((self.height*1.5)*self.scale)<Window.height: self.y += deltapos
        elif to==3 and self.y>0: self.y -= deltapos
        elif to==4 and self.scale<2.: self.scale += deltascale
        elif to==5 and self.scale>0.4: self.scale -= deltascale
        elif to==6: self.rotation -= 1
        elif to==7: self.rotation += 1
        else: r = self.app.sliders_reward[6].value  # penalty for inaction
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

    def widjet_area(self): return self.width*self.height*(self.scale/2)

    def change_emulation(self):
        self.emulation = self.set_emulation(True) if not(self.emulation) else self.set_emulation(False)

    def start_emulation(self):
        self.emulation = self.set_emulation(True)

    def stop_emulation(self):
        self.emulation = self.set_emulation(False)

    def set_emulation(self, on=False):
        method = self.IQL_adapt if self.mode == 'IQL' else self.simple_adapt
        if on:
            Clock.schedule_interval(method, 1. / 30.)
            return True
        else:
            Clock.unschedule(method)
            return False

    def on_touch_up(self, touch):
        if touch.grab_current == self: self.tap_event()
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

