# Adaptive Mobile User Interface based on Machine Learning Model (Multi-agent Reinforcement Learning Approach)
### By Dmitry Vidmanov, Alexander Alfimtsev
#### Copyright (c) 2024 All rights reserved.

This project is a mobile user interface sandbox platform based on the popular Kivy software library, providing rapid development capabilities for applications equipped with user interfaces, such as multi-touch applications.

"**Mobile User Interface Adaptation Based on Usability Reward Model and Multi-Agent Reinforcement Learning**"

https://www.mdpi.com/2414-4088/8/4/26

Discover how the research integrates usability metrics into multi-agent reinforcement learning for mobile UI adaptation. Addressing key challenges in digital product development, the study explores the effectiveness of different RL algorithms on usability metrics.

Version 0.0.3.4 (22.10.2023) MARLMUI

![img.png](data/MobAdaptUI_v34_Agents.gif)

Version 0.0.3.3 (11.10.2023) DQN via PyTorch

![img.png](data/MobAdaptUI_v33_Agents.gif)

Version 0.0.3.1 (25.04.2023)

![img.png](data/MobAdaptUI_v3_Agents.gif)

Version 0.0.2.0 (08.02.2022)

![img.png](data/interface002.png)

Version 0.0.1.6 (28.12.2021)

A simple Kivy Python application that adapts a tiled interface based on the number of clicks (a simple frequency analysis).
The number of clicks on the button is the criterion of adaptation. We adapt the interface by a simple permutation, which varies to bring the buttons with the largest number of clicks closer to one of the 4 corners of the screen. The preferred edge setting is selected using the left, right, top, bottom keys.

![img.png](data/interface0016.png)


## Android
You can build for Android using buildozer on Linux.
## Aurora OS
An application is being developed on the [Aurora OS ](https://community.omprussia.ru/documentation/platform.html) platform. 
## Install buildozer

Follow the instructions for your platform [here](https://pypi.org/project/buildozer/) 

Create a new buildozer.spec file or use the example one from the repo.
```
buildozer init
```
Make the following changes to the buildozer.spec file
```
source.include_exts = py,png,jpg,kv,atlas
requirements = python3,kivy
```
Change the architecture you are building for to match that of your device or emulator (f.e. arm64-v8a)
```
android.arch = arm64-v8a
```
Build the APK
```
buildozer android debug
```
and install it with
```
adb install bin/myapp-0.1-x86-debug.apk
```
