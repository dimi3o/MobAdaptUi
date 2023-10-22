import random
import math as m
from agent import plot
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
            return [[-0.008230124551748114, 0.3134321159927854, 0.4845808085884755, 0.9942921716447548], [0.7655172413793103, 0.7132978723404254, 0.37499999999999994, 0.000001], [0.22828552537890343, 0.3264145170401057, 0.4643659582770843, 0.012643920249283902],             [0.4298850574712644, 0.8515957446808511, 0.37499999999999994, 0.000001], [0.7518844392687961, 0.3548869179423002, 0.45301551188171785, 0.012573234346845319],             [-0.11158338630110354, -0.024367208840369945, 1, 0.9388311143034922], [0.020914892658128004, 0.23638784958449766, 0.3947129238531719, 0.9881563111314371], [0.26676771504357716, 0.19525539328341573, 0.42525899206853685, 0.9994275026739331], [0.5517331693482739, 0.8087318415211922, 0.4220226439163873, 0.011182868885693642], [0.3702017249968683, 0.14896526007510624, 0.9090558625292008, 0.020323368671318398], [0.22589754373915535, 0.6935732802605568, 0.45931725266643414, 0.9863470772187479], [0.23218390804597688, 0.8569148936170212, 0.37499999999999994, 0.000001], [0.3205391988636713, 0.7983301344318087, 0.42837396390111043, 0.9995938513851065], [0.6252702920073453, 0.7282164137705927, 0.43385023255235206, 0.004147062293050396], [0.7609195402298851, 0.7957446808510638, 0.37499999999999994, 0.000001], [0.01780656633009511, 0.4274391677442108, 0.44683826210671773, 0.014214763472146175], [0.20000000000000007, 0.26010638297872335, 0.37499999999999994, 0.000001], [0.04963281380414406, 0.17595478097616177, 0.42235826232188806, 0.9973526210867736], [0.3834296909872853, 0.7134966969061416, 0.37371318265406156, 0.01978346378700513], [0.17201600078203072, -0.03272778450873812, 1, 0.8536436235900045], [0.032183908045977046, 0.6856382978723408, 0.37499999999999994, 0.000001], [0.12244268702002026, 0.3734689010207214, 0.5243700466212033, 0.014175638680621383], [0.4116856025639025, 0.30497592513907795, 0.7385345008407359, 0.015062261841328473], [0.7924548568380725, 0.8403636583277078, 0.40729100939898294, 0.0023046071766622545], [0.44242214342571107, 0.7399411381217927, 0.4434991898106811, 0.005608919139017288], [0.5678160919540229, 0.8611702127659577, 0.37499999999999994, 0.000001], [0.02928633485720296, 0.7284850131109782, 0.4144606864642285, 0.007892396057354423], [0.4243773612147172, 0.832537749996288, 0.43681966909128944, 0.0027794956897906897], [0.14752742895632218, 0.7550044155384159, 0.4235686987598174, 0.0032753832424712304], [0.5706878090065147, 0.005411763631787168, 1, 0.9562131326830683],             [0.7286760503585811, 0.6587476750061301, 0.4205872327393562, 0.9866073091248078], [0.5166216617611323, 0.7090535913627355, 0.4247750067320609, 0.9972870664427667], [0.591445775946827, 0.23596613773943037, 0.45758279444106253, 0.9999091146203785], [0.706575123283081, 0.20338669752496627, 1, 0.9949979505211666], [0.14446331781197486, 0.7820966171323913, 0.5656269492475828, 0.9611079854478524], [0.3448275862068966, 0.41436170212765977, 0.37499999999999994, 0.000001], [0.2595186589011129, 0.7460464320346818, 0.5484779421679008, 0.9962590111949294], [0.4999293193586481, 0.6716779288683915, 0.30854379201366594, 0.9939891306459728],             [0.029885057471264385, 0.846808510638298, 0.37499999999999994, 0.000001], [-0.0008890793528547774, 0.7847135213357972, 0.4508904030990692, 0.006381545834408697]]
            # return [[0.02, 0.31, 1.07, 0.12], [0.82, 0.66, 1.07, -0.07], [0.28, 0.27, 1.33, 0.03], [0.46, 0.80, 1.08, -0.07], [0.82, 0.37, 1.01, 0.00], [0.01, 0.00, 2.21, 0.48], [0.02, 0.22, 1.00, -0.28], [0.28, 0.19, 1.10, -0.05], [0.59, 0.81, 1.02, -0.05], [0.50, 0.18, 1.52, -0.09], [0.33, 0.66, 0.94, 0.05], [0.30, 0.84, 1.03, -0.16], [0.35, 0.75, 1.02, 0.14], [0.61, 0.70, 1.05, 0.16], [0.76, 0.78, 1.05, -0.02], [0.05, 0.42, 1.11, -0.02], [0.19, 0.23, 1.01, -0.14], [0.12, 0.19, 0.97, 0.00], [0.42, 0.66, 0.98, 0.29], [0.31, 0.00, 1.99, 0.72], [0.08, 0.66, 1.09, 0.05], [0.19, 0.37, 1.00, 0.16], [0.46, 0.31, 1.56, -0.34], [0.82, 0.82, 0.93, 0.05], [0.53, 0.71, 1.00, -0.17], [0.55, 0.83, 0.87, 0.02], [0.10, 0.71, 0.86, 0.10], [0.39, 0.79, 1.10, 0.22], [0.18, 0.67, 1.05, -0.33], [0.67, 0.00, 1.92, 0.40], [0.79, 0.66, 0.97, -0.10], [0.62, 0.67, 1.00, -0.21], [0.65, 0.24, 1.08, -0.21], [0.78, 0.21, 1.43, 0.24], [0.22, 0.78, 1.08, -0.17], [0.39, 0.42, 1.03, 0.05], [0.22, 0.75, 1.09, -0.05], [0.55, 0.66, 0.96, -0.14], [0.07, 0.82, 1.06, 0.09], [0.04, 0.75, 0.88, 0.28]]
        elif name == 'eur':
            return [[0.6676319444137692, 0.21024062663108345, 0.48458080858848845, 0.9942921716447555], [0.006896551724137996, 0.7569148936170212, 0.37499999999999994, 0.000001], [0.18230851388465055, 0.7498187723592548, 0.46436595827708155, 0.012643920249284235], [0.8666666666666667, 0.8398936170212766, 0.37499999999999994, 0.000001], [0.1610798415676467, 0.3250996838997469, 0.45301551188171785, 0.012573234346845319], [0.1399892802375793, 0.24836242787582435, 0.43832785511897415, 0.9884563896803794], [0.5611447777155992, 0.07894104107385938, 0.39471292385316636, 0.9881563111314376], [0.39186641682687207, 0.7834471989068903, 0.5113127347541316, 0.9959885401918452], [0.8804597701149425, 0.847872340425532, 0.37499999999999994, 0.000001], [-0.022827123626461554, 0.3011312813819092, 0.527392782920282, 0.9953091632962525], [0.8580814517851323, 0.8893179611116205, 0.4593172526664164, 0.9863470772187469], [0.8988505747126435, 0.8601063829787234, 0.37499999999999994, 0.000001], [0.7619185092084989, 0.5408833259211704, 0.42837396390109256, 0.999593851385108], [0.4436610966050465, 0.6037483286642097, 0.4338502325523698, 0.004147062293048676], [0.1517241379310345, 0.4191489361702128, 0.37499999999999994, 0.000001], [0.7327490950657273, 0.12318384859527465, 0.44683826210671995, 0.014214763472145675], [0.9057471264367816, 0.8632978723404254, 0.37499999999999994, 0.000001], [-0.011272366980091746, 0.39516690218096207, 0.49226576007210743, 0.995016439634619], [0.15617582881107092, 0.0478363038484063, 0.49917621189638417, 0.011629325629720522], [0.01727625102267989, 0.8131454976538642, 0.5003174770490336, 0.9979502440826405], [0.8735632183908046, 0.8452127659574469, 0.37499999999999994, 0.000001], [0.7109484341464571, 0.04900081591433844, 0.5243700466212037, 0.014175638680621994], [0.3146103021807062, 0.3865820405914661, 0.4613906175142564, 0.9998868495403345], [0.5516036950077987, 0.5393985010740501, 0.4925544912914272, 0.9893758967815839], [0.5169958840683451, 0.15041832354255022, 0.5396001470787524, 0.003609543412844418], [0.9126436781609195, 0.8324468085106386, 0.37499999999999994, 0.000001], [0.6821598980755937, 0.6189105450258718, 0.4144606864642285, 0.00789239605735409], [0.08414747615724597, 0.11232498403884125, 0.4368196690912849, 0.0027794956897909673], [0.03133787555694237, 0.45108308462501223, 0.5231877235881941, 0.9965069158902211], [0.027586206896551724, 0.05957446808510635, 0.37499999999999994, 0.000001], [0.9033886940367419, 0.8172583133040023, 0.4205872327393562, 0.9866073091248078], [0.6752423514163047, 0.4771386977457141, 0.4247750067320786, 0.9972870664427633], [0.12960232626430504, 0.683719693024352, 0.5481583538777992, 0.9978648956398708], [0.23588469190753272, 0.8334392438403431, 0.4606430913839741, 0.007353735314162224], [0.8731989499958829, 0.8342242767068592, 0.5656269492475656, 0.9611079854478563], [0.8758620689655172, 0.8686170212765959, 0.37499999999999994, 0.000001], [0.41354164740686006, 0.49923792139638357, 0.5484779421679089, 0.9962590111949294], [0.9160212733816364, 0.8780609075917959, 0.30854379201366594, 0.9939891306459738], [0.9241379310344827, 0.846808510638298, 0.37499999999999994, 0.000001], [0.3186511505322027, 0.29641564899537176, 0.4508904030990738, 0.006381545834407754]]
            # return [[0.72, 0.19, 1.17, -0.06], [0.05, 0.76, 0.99, -0.00], [0.22, 0.77, 0.99, 0.02], [0.90, 0.85, 1.00, 0.02], [0.16, 0.32, 1.31, -0.03], [0.16, 0.27, 0.99, -0.02], [0.61, 0.05, 0.98, -0.02], [0.40, 0.80, 1.20, -0.04], [0.87, 0.87, 1.00, -0.02], [-0.01, 0.33, 1.16, -0.06], [0.89, 0.91, 1.01, -0.00], [0.89, 0.87, 0.99, 0.02], [0.77, 0.56, 1.27, -0.00], [0.43, 0.61, 1.25, 0.18], [0.16, 0.40, 1.29, -0.03], [0.79, 0.11, 1.00, -0.00], [0.86, 0.90, 1.00, 0.02], [0.00, 0.41, 0.99, -0.00], [0.25, 0.06, 1.00, -0.00], [0.07, 0.82, 1.35, -0.04], [0.92, 0.86, 1.00, -0.00], [0.79, 0.04, 0.98, 0.05], [0.32, 0.39, 1.57, -0.04], [0.63, 0.57, 1.00, -0.00], [0.61, 0.14, 1.00, -0.02], [0.90, 0.89, 1.00, 0.02], [0.65, 0.64, 1.32, 0.11], [0.08, 0.12, 1.00, -0.00], [0.06, 0.44, 1.35, 0.05], [0.04, 0.03, 1.42, 0.06], [0.91, 0.86, 1.00, -0.00], [0.75, 0.50, 1.00, -0.02], [0.20, 0.72, 0.99, 0.02], [0.26, 0.84, 1.01, -0.02], [0.91, 0.88, 1.00, 0.02], [0.92, 0.90, 1.00, -0.02], [0.45, 0.54, 1.33, 0.03], [0.92, 0.88, 0.97, -0.00], [0.90, 0.89, 0.99, -0.05], [0.37, 0.32, 1.01, 0.02]]
        elif name == 'asia':
            return [[0.6735632183908046, 0.09946808510638298, 0.37499999999999994, 0.000001], [0.19770114942528738, 0.11755319148936179, 0.37499999999999994, 0.000001], [0.6022988505747127, 0.09999999999999996, 0.37499999999999994, 0.000001], [0.5954022988505747, 0.10478723404255322, 0.37499999999999994, 0.000001], [0.2145593869731807, 0.710463600333751, 0.6042301805654543, 0.026378069702707008], [0.5975765911360872, 0.05204146903585885, 0.5461593732577027, 0.9706468483123682], [0.6137931034482759, 0.09574468085106401, 0.37499999999999994, 0.000001], [0.696551724137931, 0.08829787234042537, 0.37499999999999994, 0.000001], [0.632183908045977, 0.12659574468085114, 0.37499999999999994, 0.000001], [0.639080459770115, 0.11542553191489367, 0.37499999999999994, 0.000001], [0.5659838482245874, 0.6554306730997387, 1, 0.013834875926115653], [0.2091954022988506, 0.3281914893617021, 0.37499999999999994, 0.000001], [0.7172413793103448, 0.12180851063829781, 0.37499999999999994, 0.000001], [0.6459770114942529, 0.10000000000000009, 0.37499999999999994, 0.000001], [0.17241379310344832, 0.18510638297872342, 0.37499999999999994, 0.000001], [0.07586206896551724, 0.7675531914893617, 0.37499999999999994, 0.000001], [0.6505747126436782, 0.0537234042553191, 0.37499999999999994, 0.000001], [0.17956675508399633, 0.8124631751227483, 0.8589263041685538, 0.9941336039059276], [0.5379310344827586, 0.09308510638297883, 0.37499999999999994, 0.000001], [0.6114942528735632, 0.12978723404255316, 0.37499999999999994, 0.000001], [0.5724137931034483, 0.052659574468085245, 0.37499999999999994, 0.000001], [0.05057471264367816, 0.8276595744680851, 0.37499999999999994, 0.000001], [0.041379310344827586, 0.35478723404255336, 0.37499999999999994, 0.000001], [0.6275862068965518, 0.1090425531914894, 0.37499999999999994, 0.000001], [0.6298850574712643, 0.1058510638297872, 0.37499999999999994, 0.000001], [0.7425287356321839, 0.08244680851063836, 0.37499999999999994, 0.000001], [0.04367816091954024, 0.15797872340425528, 0.37499999999999994, 0.000001], [0.3333333333333333, 0.18829787234042542, 0.37499999999999994, 0.000001], [0.19540229885057472, 0.2425531914893617, 0.37499999999999994, 0.000001], [0.6620689655172414, 0.1191489361702127, 0.37499999999999994, 0.000001], [0.524304332018905, 0.4406364859967904, 1, 0.004300438052716016], [0.6574712643678161, 0.10372340425531915, 0.37499999999999994, 0.000001], [0.5885057471264368, 0.10265957446808505, 0.37499999999999994, 0.000001], [0.011494252873563218, 0.248936170212766, 0.37499999999999994, 0.000001], [0.48834422075720096, 0.5002228460887295, 1, 0.014361369871126795], [0.6160919540229886, 0.10053191489361711, 0.37499999999999994, 0.000001], [0.34942528735632183, 0.29574468085106376, 0.37499999999999994, 0.000001], [0.039112660478531104, 0.6521976313266121, 0.6403196062190399, 0.0047636949875737855], [0.367816091954023, 0.23404255319148948, 0.37499999999999994, 0.000001], [0.4107142641199126, 0.2705418200946872, 1, 0.9726486787242818]]
            # return [[0.67, 0.07, 1.00, -0.00], [0.22, 0.10, 1.31, 0.04], [0.73, 0.10, 1.00, -0.00], [0.59, 0.04, 1.00, -0.00], [0.24, 0.74, 1.33, -0.16], [0.56, 0.08, 1.00, -0.00], [0.75, 0.12, 1.00, -0.00], [0.70, 0.08, 1.00, -0.00], [0.68, 0.08, 1.00, -0.00], [0.65, 0.09, 1.00, -0.00], [0.63, 0.65, 2.11, 0.07], [0.22, 0.33, 1.00, -0.00], [0.73, 0.08, 1.00, -0.00], [0.69, 0.03, 1.00, -0.00], [0.20, 0.18, 1.00, -0.00], [0.08, 0.77, 1.17, -0.04], [0.68, 0.06, 1.00, -0.00], [0.22, 0.82, 1.65, 0.07], [0.60, 0.09, 1.00, -0.00], [0.66, 0.07, 1.00, -0.00], [0.64, 0.04, 1.00, -0.00], [0.09, 0.85, 1.00, -0.00], [0.07, 0.33, 1.00, -0.00], [0.75, 0.10, 1.00, -0.00], [0.73, 0.05, 1.00, -0.00], [0.80, 0.07, 1.00, -0.00], [-0.01, 0.13, 1.26, 0.11], [0.34, 0.17, 1.32, -0.02], [0.20, 0.24, 1.00, -0.00], [0.74, 0.09, 1.00, -0.00], [0.62, 0.46, 2.07, -0.09], [0.70, 0.11, 1.00, -0.00], [0.63, 0.10, 1.00, -0.00], [-0.02, 0.23, 1.29, 0.01], [0.58, 0.53, 2.70, -0.11], [0.67, 0.06, 1.00, -0.00], [0.33, 0.28, 1.20, 0.08], [0.02, 0.67, 1.49, -0.08], [0.39, 0.23, 1.00, -0.00], [0.44, 0.26, 3.25, 0.28]]

class FlyScatterV3(Scatter):#(TouchRippleBehavior, Scatter):
    velocity = ListProperty([2, 1])
    emulation = BooleanProperty(False)
    mode = 'Fly adapt'
    app = None
    memory = None
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
    nx, ny, ns, nr = 0, 0, 0, 0
    taps = 0
    steps_learning = 0
    vect_state = []

    def __init__(self, **kwargs):
        super(FlyScatterV3, self).__init__(**kwargs)
        self.velocity[0] *= random.choice([-2, 2])
        self.velocity[1] *= random.choice([-2, 2])
        self.text = 'flyscatter'
        self.color = random.choice(allcolors)
        if self.env: self.set_vect_state()

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

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.app.TAU + target_net_state_dict[key] * (1 - self.app.TAU)
        self.target_net.load_state_dict(target_net_state_dict)
        

        # Синхронизируем веса основной и целевой нейронной сети каждые target_update шагов
        # if self.env.steps_left % self.app.target_update == 0:
        #     self.target_net.load_state_dict(self.policy_net.state_dict())
        #     # print('--- target_net update ---')
        return r

    def set_vect_state(self):
        nx = (self.x - 0) / (Window.width - 0)
        ny = (self.y - 0) / (Window.height - 0)
        ns = min(1, max(0, (self.scale - 0.4) / (2 - 0.4)))
        nr = ((self.rotation-180)/180 + 1) / 2
        self.vect_state = [int(self.id), int(self.taps), nx, ny, ns, nr]
        self.env.vect_state = self.vect_state
        return self.vect_state

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
        if self.mode ==  'Rotate adapt' or self.mode == 'Fly+Size+Rotate adapt': self.rotation += random.choice([-1, 1])
        elif self.mode == 'Fly adapt' or self.mode == 'Fly+Size+Rotate adapt':
            self.x += self.deltaposxy*self.velocity[0]
            self.y += self.deltaposxy*self.velocity[1]
            if self.x < 0 or (self.x + 2*self.width//3) > Window.width:
                self.velocity[0] *= -1
            if self.y < 0 or (self.y + 2*self.height//3) > Window.height:
                self.velocity[1] *= -1
        elif self.mode == 'Size adapt' or self.mode == 'Fly+Size+Rotate adapt':
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

