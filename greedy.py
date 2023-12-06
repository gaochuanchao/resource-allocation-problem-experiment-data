#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 5/6/2023 6:24 PM
# @Author  : Gao Chuanchao
# @Email   : jerrygao53@gmail.com
# @File    : greedy.py
"""
The Zero-Slack Greedy Algorithm
"""
import numpy as np
import cloud
import math
from config import *


def find_all_combinations(cs: cloud.Cloud, ts: cloud.TaskSet):
    # find all feasible combinations
    comb_all = []
    for i in range(ts.t_count):
        # obtain the access points that task i can be offloaded to
        task_links = ts.task_ap_link[i]
        ap_i = list(np.where(task_links == 1)[0])
        for j in ap_i:
            for k in range(cs.s_count):
                comb_all.append((i, j, k))

    return comb_all


def calc_offload_energy(size, b_ij, offload_t):
    log_term = size / (offload_t * b_ij)
    if log_term > 10:
        return -1
    power = (math.pow(2, log_term) - 1) * NOISE / CHANNEL_GAIN
    energy = power * offload_t

    if power > MAX_OFF_POWER:  # cap transmission power at 0.1W
        return -1
    else:
        return energy


def resource_allocation(cs: cloud.Cloud, ts: cloud.TaskSet, all_comb: list, alpha):
    comm_res = {}
    comp_res = {}
    ratio_b = {}
    ratio_c = {}
    saved_energy = {}
    feasi_comb = set()

    log_term = math.log2(0.1 * CHANNEL_GAIN / NOISE + 1)    # assume always use the maximum offloading power

    for comb in all_comb:  # comb = (i,j,k)
        if ts.deadline[comb[0]] > cs.delta[comb[1], comb[2]]:
            numer = ts.size[comb[0]] / (cs.ap_cap[comb[1]] * log_term)
            denom = ts.cycle[comb[0]] / cs.s_cap[comb[2]]
            ratio = numer / denom
            gamma_ijk = ratio / (1 + ratio)  # gamma_ijk/(1- gamma_ijk) = ratio

            process_time = (1 - gamma_ijk) * (ts.deadline[comb[0]] - cs.delta[comb[1], comb[2]])
            offload_time = gamma_ijk * (ts.deadline[comb[0]] - cs.delta[comb[1], comb[2]])
            b_ijk = ts.size[comb[0]] / (offload_time * log_term)
            c_ijk = ts.cycle[comb[0]] / process_time
            b_ijk = max(b_ijk, 0.1)     # minimum bandwidth 0.1 MHz
            c_ijk = max(c_ijk, 1)
            r_b = b_ijk / cs.ap_cap[comb[1]]
            r_c = c_ijk / cs.s_cap[comb[2]]

            if (r_b <= alpha) and (r_c <= alpha):
                if offload_time > 0:
                    off_energy = 0.1 * offload_time
                    saved_e = ts.local_energy[comb[0]] - off_energy
                    if saved_e > 0:
                        comm_res[comb] = b_ijk
                        comp_res[comb] = c_ijk
                        ratio_b[comb] = r_b
                        ratio_c[comb] = r_c
                        saved_energy[comb] = saved_e
                        feasi_comb.add(comb)
            elif (r_b <= alpha) and (r_c > alpha):  # need decrease gamma_ijk
                while gamma_ijk > 0.05:
                    gamma_ijk = gamma_ijk - 0.05  # decrease 0.05 every time

                    process_time = (1 - gamma_ijk) * (ts.deadline[comb[0]] - cs.delta[comb[1], comb[2]])
                    offload_time = gamma_ijk * (ts.deadline[comb[0]] - cs.delta[comb[1], comb[2]])
                    b_ijk = ts.size[comb[0]] / (offload_time * log_term)
                    c_ijk = ts.cycle[comb[0]] / process_time
                    b_ijk = max(b_ijk, 0.1)
                    c_ijk = max(c_ijk, 1)
                    r_b = b_ijk / cs.ap_cap[comb[1]]
                    r_c = c_ijk / cs.s_cap[comb[2]]

                    if (r_b <= alpha) and (r_c <= alpha):
                        if offload_time > 0:
                            off_energy = 0.1 * offload_time
                            saved_e = ts.local_energy[comb[0]] - off_energy
                            if saved_e > 0:
                                comm_res[comb] = b_ijk
                                comp_res[comb] = c_ijk
                                ratio_b[comb] = r_b
                                ratio_c[comb] = r_c
                                saved_energy[comb] = saved_e
                                feasi_comb.add(comb)
                    break
            elif (r_b > alpha) and (r_c <= alpha):  # need increase gamma_ijk
                while gamma_ijk < 0.95:
                    gamma_ijk = gamma_ijk + 0.05  # increase 0.05 every time

                    process_time = (1 - gamma_ijk) * (ts.deadline[comb[0]] - cs.delta[comb[1], comb[2]])
                    offload_time = gamma_ijk * (ts.deadline[comb[0]] - cs.delta[comb[1], comb[2]])
                    b_ijk = ts.size[comb[0]] / (offload_time * log_term)
                    c_ijk = ts.cycle[comb[0]] / process_time
                    b_ijk = max(b_ijk, 0.1)
                    c_ijk = max(c_ijk, 1)
                    r_b = b_ijk / cs.ap_cap[comb[1]]
                    r_c = c_ijk / cs.s_cap[comb[2]]

                    if (r_b <= alpha) and (r_c <= alpha):
                        if offload_time > 0:
                            off_energy = 0.1 * offload_time
                            saved_e = ts.local_energy[comb[0]] - off_energy
                            if saved_e > 0:
                                comm_res[comb] = b_ijk
                                comp_res[comb] = c_ijk
                                ratio_b[comb] = r_b
                                ratio_c[comb] = r_c
                                saved_energy[comb] = saved_e
                                feasi_comb.add(comb)
                    break

    return feasi_comb, comm_res, comp_res, ratio_b, ratio_c, saved_energy


def resource_allocation3(cs: cloud.Cloud, ts: cloud.TaskSet, all_comb: list, alpha):
    """
    always use maximum bandwidth allocation
    """
    comm_res = {}
    comp_res = {}
    ratio_b = {}
    ratio_c = {}
    saved_energy = {}
    feasi_comb = set()

    log_term = math.log2(0.1 * CHANNEL_GAIN / NOISE + 1)    # when offloading power=0.1W

    for comb in all_comb:  # comb = (i,j,k)
        if ts.deadline[comb[0]] > cs.delta[comb[1], comb[2]]:
            numer = ts.size[comb[0]] / (cs.ap_cap[comb[1]] * log_term)
            denom = ts.cycle[comb[0]] / cs.s_cap[comb[2]]
            ratio = numer / denom
            gamma_ijk = ratio / (1 + ratio)  # gamma_ijk/(1- gamma_ijk) = ratio

            process_time = (1 - gamma_ijk) * (ts.deadline[comb[0]] - cs.delta[comb[1], comb[2]])
            offload_time = gamma_ijk * (ts.deadline[comb[0]] - cs.delta[comb[1], comb[2]])
            b_ijk = alpha * cs.ap_cap[comb[1]]  # always use maximum bandwidth allocation
            r_b = alpha
            c_ijk = ts.cycle[comb[0]] / process_time
            c_ijk = max(c_ijk, 1)
            r_c = c_ijk / cs.s_cap[comb[2]]

            if r_c <= alpha:
                if offload_time > 0:
                    off_energy = calc_offload_energy(ts.size[comb[0]], b_ijk, offload_time)
                    if off_energy > 0:
                        saved_e = ts.local_energy[comb[0]] - off_energy
                        if saved_e > 0:
                            comm_res[comb] = b_ijk
                            comp_res[comb] = c_ijk
                            ratio_b[comb] = r_b
                            ratio_c[comb] = r_c
                            saved_energy[comb] = saved_e
                            feasi_comb.add(comb)
                    else:
                        r_ijk = alpha * cs.ap_cap[comb[1]] * log_term
                        offload_time = ts.size[comb[0]] / r_ijk
                        if (offload_time + cs.delta[comb[1], comb[2]]) < ts.deadline[comb[0]]:
                            process_time = ts.deadline[comb[0]] - offload_time - cs.delta[comb[1], comb[2]]
                            c_ijk = ts.cycle[comb[0]] / process_time
                            c_ijk = max(c_ijk, 1)
                            r_c = c_ijk / cs.s_cap[comb[2]]
                            if r_c <= alpha:
                                off_energy = 0.1 * offload_time
                                saved_e = ts.local_energy[comb[0]] - off_energy
                                if saved_e > 0:
                                    comm_res[comb] = b_ijk
                                    comp_res[comb] = c_ijk
                                    ratio_b[comb] = r_b
                                    ratio_c[comb] = r_c
                                    saved_energy[comb] = saved_e
                                    feasi_comb.add(comb)
            else:
                r_c = alpha
                c_ijk = alpha * cs.s_cap[comb[2]]
                process_time = ts.cycle[comb[0]] / c_ijk
                if (process_time + cs.delta[comb[1], comb[2]]) < ts.deadline[comb[0]]:
                    offload_time = ts.deadline[comb[0]] - process_time - cs.delta[comb[1], comb[2]]
                    off_energy = calc_offload_energy(ts.size[comb[0]], b_ijk, offload_time)
                    if off_energy > 0:
                        saved_e = ts.local_energy[comb[0]] - off_energy
                        if saved_e > 0:
                            comm_res[comb] = b_ijk
                            comp_res[comb] = c_ijk
                            ratio_b[comb] = r_b
                            ratio_c[comb] = r_c
                            saved_energy[comb] = saved_e
                            feasi_comb.add(comb)

    return feasi_comb, comm_res, comp_res, ratio_b, ratio_c, saved_energy


def resource_allocation2(cs: cloud.Cloud, ts: cloud.TaskSet, all_comb: list, alpha):
    comm_res = {}
    comp_res = {}
    ratio_b = {}
    ratio_c = {}
    saved_energy = {}
    feasi_comb = set()

    log_term = math.log2(0.1 * CHANNEL_GAIN / NOISE + 1)    # assume always use the maximum offloading power

    for comb in all_comb:  # comb = (i,j,k)
        if ts.deadline[comb[0]] > cs.delta[comb[1], comb[2]]:
            numer = ts.size[comb[0]] / cs.ap_cap[comb[1]]
            denom = ts.cycle[comb[0]] / cs.s_cap[comb[2]]
            ratio = numer / denom
            gamma_ijk = ratio / (1 + ratio)  # gamma_ijk/(1- gamma_ijk) = ratio

            process_time = (1 - gamma_ijk) * (ts.deadline[comb[0]] - cs.delta[comb[1], comb[2]])
            offload_time = gamma_ijk * (ts.deadline[comb[0]] - cs.delta[comb[1], comb[2]])
            b_ijk = ts.size[comb[0]] / (offload_time * log_term)
            c_ijk = ts.cycle[comb[0]] / process_time
            b_ijk = max(b_ijk, 0.1)     # minimum bandwidth 0.1 MHz
            c_ijk = max(c_ijk, 1)
            r_b = b_ijk / cs.ap_cap[comb[1]]
            r_c = c_ijk / cs.s_cap[comb[2]]

            if offload_time > 0:
                off_energy = 0.1 * offload_time
                saved_e = ts.local_energy[comb[0]] - off_energy
                if saved_e > 0:
                    comm_res[comb] = b_ijk
                    comp_res[comb] = c_ijk
                    ratio_b[comb] = r_b
                    ratio_c[comb] = r_c
                    saved_energy[comb] = saved_e
                    feasi_comb.add(comb)

    return feasi_comb, comm_res, comp_res, ratio_b, ratio_c, saved_energy


def calculate_priority(feasi_comb: set, ratio_b: dict, ratio_c: dict, energy: dict):
    priority = dict()
    for comb in feasi_comb:  # comb = (i,j,k)
        p_ijk = energy[comb] / (ratio_b[comb] * ratio_c[comb])
        combs = priority.setdefault(p_ijk, set())
        combs.add(comb)

    return priority


def zero_slack_greedy(cs: cloud.Cloud, ts: cloud.TaskSet, alpha):
    """
    Zero Slack Greedy Algorithm from GLOBECOM paper
    :param cs: edge computing system
    :param ts: taskset for exam
    :param alpha: resource allocation bound
    :return:
    """
    # start_time = time.time()
    # determine all combination
    all_comb = find_all_combinations(cs, ts)

    # allocate resource to each combination
    feasi_comb, comm_res, comp_res, ratio_b, ratio_c, energy = resource_allocation3(cs, ts, all_comb, alpha)

    # calculate priority
    priority = calculate_priority(feasi_comb, ratio_b, ratio_c, energy)

    matching = []
    comm_resource = {}
    comp_resource = {}
    ap_capacity = np.copy(cs.ap_cap)
    s_capacity = np.copy(cs.s_cap)
    mapped_task = set()

    total_saved_e = 0
    p_values = sorted(priority.keys(), reverse=True)
    for p in p_values:
        for comb in priority[p]:    # comb = (i,j,k)
            if comb[0] not in mapped_task:
                if (comm_res[comb] <= ap_capacity[comb[1]]) and (comp_res[comb] <= s_capacity[comb[2]]):
                    matching.append(comb)
                    total_saved_e += energy[comb]
                    comm_resource[comb] = comm_res[comb]
                    comp_resource[comb] = comp_res[comb]

                    ap_capacity[comb[1]] = ap_capacity[comb[1]] - comm_res[comb]
                    s_capacity[comb[2]] = s_capacity[comb[2]] - comp_res[comb]
                    mapped_task.add(comb[0])

    # end_time = time.time()
    # print("ZSG total spends %s seconds" % (end_time - start_time))
    return {"mapping": matching, "energy": total_saved_e, "comm_res": comm_resource, "comp_res": comp_resource}
