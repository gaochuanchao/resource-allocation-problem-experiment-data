#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 5/2/2023 10:39 AM
# @Author  : Gao Chuanchao
# @Email   : jerrygao53@gmail.com
# @File    : cloud.py
import math
import random
import numpy as np
import utils
from config import *


class Cloud:
    """
    Edge-Computing System class.
    """

    def __init__(self):
        """
        Initialize the edge-computing system.
        Attributes:
            s_count     number of servers
            ap_count    number of access point
            s_cap       computation resource capacity of all servers
            ap_cap      bandwidth resource of all access points
            delta       data transmission delay in the backhaul network
        """
        self.unit_b = 1
        self.unit_c = 1
        self.s_count = 0
        self.ap_count = 0
        self.s_cap = np.zeros(0)
        self.ap_cap = np.zeros(0)
        self.delta = np.zeros((0,0))
        self.base = 1

    def system_units_init(self, sc, apc, s_range, ap_option, d_range, base):
        """
        Initialize system related parameters
        Here we assume an edge server and an access point with the same index are co-deployed
            :param sc: number of servers
            :param apc: number of access points
            :param s_range: [lb, ub], the range of possible server computation resource capacity
            :param ap_option: (lb, ub), the possible values of access point bandwidth capacity
            :param d_range: (lb, ub), range of delta values
            :param base: base value for resource discretization
        """
        self.base = base
        self.s_count = sc
        self.ap_count = apc

        # set capacities
        self.s_cap = np.zeros(self.s_count)
        for s in range(self.s_count):
            self.s_cap[s] = random.uniform(s_range[0], s_range[1])

        self.ap_cap = np.zeros(self.ap_count)
        for ap in range(self.ap_count):
            self.ap_cap[ap] = random.choice(ap_option)

        # set backhaul network delays
        self.delta = np.zeros((self.ap_count, self.s_count))
        for ap in range(self.ap_count):
            for s in range(self.s_count):
                if ap == s:  # zero latency for co-deployed access point and server
                    self.delta[ap, s] = 0
                else:
                    self.delta[ap, s] = random.uniform(d_range[0], d_range[1])


class TaskSet:
    """
    Taskset used for edge-computing system, including a group of tasks.
    Each task has a task size, CPU cycles demanding, profit, deadline
    """

    def __init__(self, taskset_id, count, prop):
        """
        Initialize tasksets.
        Attributes:
            id              taskset id
            t_count         number of all tasks in this taskset
            size            data size of all tasks in this taskset
            cycle           CPU cycles demanding of all tasks in this taskset
            deadline        deadline of all tasks in this taskset
            degree          default 2, the maximum number of access points a task can be offloaded to
            task_ap_link    a 2D array, define the connection between tasks and access points
            prop            resource density of the taskset, i.e., "low,low"
        """
        self.id = taskset_id
        self.t_count = count
        self.size = np.zeros(self.t_count)
        self.cycle = np.zeros(self.t_count)
        self.deadline = np.zeros(self.t_count)
        self.degree = 3
        self.task_ap_link = np.zeros((0, 0), dtype=np.uintc)
        self.prop = prop
        self.task_ub = np.zeros(self.t_count)
        self.task_uc = np.zeros(self.t_count)
        self.device_cap = np.zeros(self.t_count)
        self.local_energy = np.zeros(self.t_count)

    def generation(self, cloud: Cloud, ub, uc):
        """
        Generate taskset with given size for the cloud system.
            :param cloud: the cloud system
            :param ub: normalized bandwidth demand
            :param uc: normalized computation resource demand
        """
        # define task-ap link
        random.seed()
        self.task_ap_link = np.zeros((self.t_count, cloud.ap_count), dtype=np.uintc)
        self.link_task_ap(cloud.ap_count)

        # define task data size and CPU cycles
        self.set_data_size(TASK_DATA_SIZE, T_DENSITY)

        # set device capacity
        self.set_device_capacity(DEVICE_CAP)

        # define task deadlines
        # min_ap_cap = np.min(cloud.ap_cap)
        # min_s_cap = np.min(cloud.s_cap)
        # ub_tasks = utils.StaffordRandFixedSum(self.t_count, ub / alpha, 1)[0]
        # ub_tasks = ub_tasks * alpha * min_ap_cap
        # self.task_ub = ub_tasks
        # uc_tasks = utils.StaffordRandFixedSum(self.t_count, uc / alpha, 1)[0]
        # uc_tasks = uc_tasks * alpha * min_s_cap
        # self.task_uc = uc_tasks

        # test ============= define task deadlines
        min_ap_cap = np.min(cloud.ap_cap)
        min_s_cap = np.min(cloud.s_cap)
        ub_tasks = utils.StaffordRandFixedSum(self.t_count, ub, 1)[0]
        ub_tasks = ub_tasks * min_ap_cap
        self.task_ub = ub_tasks
        uc_tasks = utils.StaffordRandFixedSum(self.t_count, uc, 1)[0]
        uc_tasks = uc_tasks * min_s_cap
        self.task_uc = uc_tasks
        # end test =========

        self.set_deadline(ub_tasks, uc_tasks)

        # calculate the local processing energy consumption
        self.get_local_energy()

    def link_task_ap(self, ap_count: int):
        """
        Define the connection between tasks and access points
            :param ap_count: number of access points
        """
        for i in range(self.t_count):
            # define how many access points each task can be offloaded to
            task_cardinality = random.randint(2, self.degree)
            for count in range(task_cardinality):
                ap = normal_choice(range(ap_count))
                # define the linkage between tasks and access points
                self.task_ap_link[i, ap] = 1

    def set_data_size(self, size_range, density):
        """
        Sample task data size, and calculate its CPU cycles
        :param size_range: the sample range of task data size
        :param density: number of cycles per bit
        """
        for i in range(self.t_count):
            self.size[i] = random.uniform(size_range[0], size_range[1])
            self.cycle[i] = self.size[i] * density

    def set_device_capacity(self, cap_range):
        """
        Sample device capacity in Mega Cycles/s
        :param cap_range: the sample range of task data size
        """
        for i in range(self.t_count):
            self.device_cap[i] = random.uniform(cap_range[0], cap_range[1])

    def set_deadline(self, ub_tasks, uc_tasks):
        """
        Set deadline for each task in this taskset.
            :param ub_tasks: the potential ap resource demand
            :param uc_tasks: the potential server resource demand
        """
        log_term = math.log2(MAX_OFF_POWER * CHANNEL_GAIN / NOISE + 1)  # assume use the maximum offloading power
        for i in range(self.t_count):
            # the minimum allocate resource are set at 0.1MHz and 1 Mega cycles/s
            # this avoids the StaffordRandFixedSum generating too small values
            t_offload = self.size[i] / (max(ub_tasks[i], 0.1) * log_term)
            t_process = self.cycle[i] / max(uc_tasks[i], 1)
            min_time = self.cycle[i] / self.device_cap[i]
            dl = min_time
            count = 50
            while True:
                if count > 0:
                    count -= 1
                    delta = random.gauss(0.008, 0.003)      # mean 8ms, std 3ms
                    if delta >= 0:
                        total_time = t_offload + delta + t_process
                        if total_time >= dl:
                            dl = total_time
                            break
                else:
                    break
            self.deadline[i] = dl

    def get_local_energy(self):
        """
        Calculate the energy consumed for local processing
        """
        for i in range(self.t_count):
            # power = kf^2 (power per CPU cycles)
            self.local_energy[i] = POWER_EFI * 1e+12 * self.device_cap[i] * self.device_cap[i] * self.cycle[i]


def normal_choice(lst, mean=None, stddev=None):
    """
    Select one element from a list following the normal distribution
    :param lst:
    :param mean:
    :param stddev:
    :return:
    """
    if mean is None:
        # if mean is not specified, use center of list
        mean = (len(lst) - 1) / 2

    if stddev is None:
        # if stddev is not specified, let list be -3 ... +3 standard deviations
        stddev = len(lst) / 6

    while True:
        index = int(random.normalvariate(mean, stddev) + 0.5)
        if 0 <= index < len(lst):
            return lst[index]

