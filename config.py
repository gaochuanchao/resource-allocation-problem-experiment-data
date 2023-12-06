#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 5/2/2023 4:07 PM
# @Author  : Gao Chuanchao
# @Email   : jerrygao53@gmail.com
# @File    : config.py
"""
Define task and edge-cloud system parameters
"""

# ======= Combinations ======= #
BASE = 1.1
ALPHA = (1/16, 1/12, 1/6)
# I_COUNT = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
LOW_R = [0.7, 1.0]
# UPPER_R = [1.0, 1.3]
UPPER_R = [1.2, 1.5]
# TS_COUNT = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
TS_RANGE = [50, 200]
REPEAT_R = 30   # sample 50 different ranges for each range combination
REPEAT_SIZE = 30    # sample 30 taskset sizes for each sampled range

# ======= Tasks ======= #
TASK_DATA_SIZE = [0.1, 0.2]    # Mb
T_DENSITY = 150    # 15~30 Mega Cycles
UNIT_B = 1     # Mbps
# UNIT_C = 100    # M-cycles/s
UNIT_C = 50    # M-cycles/s


# ======= End Device ======= #
CHANNEL_GAIN = 1e-5   # 50 dB
NOISE = 8e-8   # square of channel noise power
# Expected transmission power 0.01W to 0.1W
DEVICE_CAP = [1000, 2000]   # Mega Cycles/s
POWER_EFI = 1e-28   # energy consumption coefficient
MAX_OFF_POWER = 0.1  # maximum offloading power, 0.1W

# ======= Access Point ======= #
# AP_CAPACITY = (1200, 2400)  # Mbps, 802.11ax (Wi-Fi 6) protocol
B_CAPACITY = (80, 120)   # MHz, 802.11n protocol
AP_COUNT = 12
DELTA_RANGE = [0.003, 0.03]   # 3ms ~ 30ms

# ======= Server ======= #
S_CAPACITY = [20000, 30000]   # Mega Cycles/s (10~60 Giga Cycles/s)
S_COUNT = 15


