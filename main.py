#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 5/2/2023 10:36 AM
# @Author  : Gao Chuanchao
# @Email   : jerrygao53@gmail.com
# @File    : main.py
import math
import random
import numpy as np
import cloud
import utils
from config import *
import approximation as ap
import time
import greedy as gd
import ldm


def model_generation():
    # define edge computing system
    edge_sys = cloud.Cloud()
    # def system_units_init(self, sc, apc, s_range, ap_option, d_range, base)
    edge_sys.system_units_init(S_COUNT, AP_COUNT, S_CAPACITY, B_CAPACITY, DELTA_RANGE, BASE)
    # write data to local files
    utils.write(edge_sys, "data/cloud_system")


# def create_tasksets():
#     edge_sys: cloud.Cloud = utils.read("data/cloud_system.pickle")
#     print("ub_total: ", np.sum(edge_sys.ap_cap)/np.min(edge_sys.ap_cap))
#     print("uc_total", np.sum(edge_sys.s_cap)/np.min(edge_sys.s_cap))
#     t_id = 0
#     alpha_index = 0
#     for alp in ALPHA:
#         t_id_start = t_id
#         t_sets = []
#         for count in range(REPEAT):
#             ub_ratio = random.uniform(LOW_R[0], LOW_R[1])
#             uc_ratio = random.uniform(LOW_R[0], LOW_R[1])
#             ub = ub_ratio * np.sum(edge_sys.ap_cap)/np.min(edge_sys.ap_cap)
#             uc = uc_ratio * np.sum(edge_sys.s_cap)/np.min(edge_sys.s_cap)
#             # min_ts = math.ceil(max(ub / alpha, uc / alpha))
#             # SIZE = [min_ts + r*50 for r in range(5)]
#             # min_ts = max(ub / alpha, uc / alpha)
#             # SIZE = [math.ceil(min_ts * r) for r in TS_COUNT]
#             for size in TS_COUNT:
#                 t_set = cloud.TaskSet(t_id, size, "low,low")
#                 t_set.generation(edge_sys, TASK_DATA_SIZE, T_DENSITY, alp, ub, uc)
#                 t_id += 1
#                 t_sets.append(t_set)
#
#         for count in range(REPEAT):
#             ub_ratio = random.uniform(LOW_R[0], LOW_R[1])
#             uc_ratio = random.uniform(UPPER_R[0], UPPER_R[1])
#             ub = ub_ratio * np.sum(edge_sys.ap_cap) / np.min(edge_sys.ap_cap)
#             uc = uc_ratio * np.sum(edge_sys.s_cap) / np.min(edge_sys.s_cap)
#             # min_ts = math.ceil(max(ub / alpha, uc / alpha))
#             # SIZE = [min_ts + r * 50 for r in range(5)]
#             for size in TS_COUNT:
#                 t_set = cloud.TaskSet(t_id, size, "low,high")
#                 t_set.generation(edge_sys, TASK_DATA_SIZE, T_DENSITY, alp, ub, uc)
#                 t_id += 1
#                 t_sets.append(t_set)
#
#         for count in range(REPEAT):
#             ub_ratio = random.uniform(UPPER_R[0], UPPER_R[1])
#             uc_ratio = random.uniform(LOW_R[0], LOW_R[1])
#             ub = ub_ratio * np.sum(edge_sys.ap_cap) / np.min(edge_sys.ap_cap)
#             uc = uc_ratio * np.sum(edge_sys.s_cap) / np.min(edge_sys.s_cap)
#             # min_ts = math.ceil(max(ub / alpha, uc / alpha))
#             # SIZE = [min_ts + r * 50 for r in range(5)]
#             for size in TS_COUNT:
#                 t_set = cloud.TaskSet(t_id, size, "high,low")
#                 t_set.generation(edge_sys, TASK_DATA_SIZE, T_DENSITY, alp, ub, uc)
#                 t_id += 1
#                 t_sets.append(t_set)
#
#         for count in range(REPEAT):
#             ub_ratio = random.uniform(UPPER_R[0], UPPER_R[1])
#             uc_ratio = random.uniform(UPPER_R[0], UPPER_R[1])
#             ub = ub_ratio * np.sum(edge_sys.ap_cap) / np.min(edge_sys.ap_cap)
#             uc = uc_ratio * np.sum(edge_sys.s_cap) / np.min(edge_sys.s_cap)
#             # min_ts = math.ceil(max(ub / alpha, uc / alpha))
#             # SIZE = [min_ts + r * 50 for r in range(5)]
#             for size in TS_COUNT:
#                 t_set = cloud.TaskSet(t_id, size, "high,high")
#                 t_set.generation(edge_sys, TASK_DATA_SIZE, T_DENSITY, alp, ub, uc)
#                 t_id += 1
#                 t_sets.append(t_set)
#
#         t_id_end = t_id
#         utils.write(t_sets, "data/dataset_"+str(t_id_start)+"_"+str(t_id_end)+"_alpha_"+str(alpha_index))
#         alpha_index += 1


def create_tasksets2():
    edge_sys: cloud.Cloud = utils.read("data/cloud_system.pickle")
    print("ub_total: ", np.sum(edge_sys.ap_cap)/np.min(edge_sys.ap_cap))
    print("uc_total", np.sum(edge_sys.s_cap)/np.min(edge_sys.s_cap))
    t_id = 0

    t_id_start = t_id
    t_sets = []
    for count in range(REPEAT_R):
        ub_ratio = random.uniform(LOW_R[0], LOW_R[1])
        uc_ratio = random.uniform(LOW_R[0], LOW_R[1])
        ub = ub_ratio * np.sum(edge_sys.ap_cap)/np.min(edge_sys.ap_cap)
        uc = uc_ratio * np.sum(edge_sys.s_cap)/np.min(edge_sys.s_cap)
        for count_s in range(REPEAT_SIZE):
            size = random.randint(TS_RANGE[0], TS_RANGE[1])
            t_set = cloud.TaskSet(t_id, size, "low,low")
            t_set.generation(edge_sys, ub, uc)
            t_id += 1
            t_sets.append(t_set)

    for count in range(REPEAT_R):
        ub_ratio = random.uniform(LOW_R[0], LOW_R[1])
        uc_ratio = random.uniform(UPPER_R[0], UPPER_R[1])
        ub = ub_ratio * np.sum(edge_sys.ap_cap) / np.min(edge_sys.ap_cap)
        uc = uc_ratio * np.sum(edge_sys.s_cap) / np.min(edge_sys.s_cap)
        for count_s in range(REPEAT_SIZE):
            size = random.randint(TS_RANGE[0], TS_RANGE[1])
            t_set = cloud.TaskSet(t_id, size, "low,high")
            t_set.generation(edge_sys, ub, uc)
            t_id += 1
            t_sets.append(t_set)

    for count in range(REPEAT_R):
        ub_ratio = random.uniform(UPPER_R[0], UPPER_R[1])
        uc_ratio = random.uniform(LOW_R[0], LOW_R[1])
        ub = ub_ratio * np.sum(edge_sys.ap_cap) / np.min(edge_sys.ap_cap)
        uc = uc_ratio * np.sum(edge_sys.s_cap) / np.min(edge_sys.s_cap)
        for count_s in range(REPEAT_SIZE):
            size = random.randint(TS_RANGE[0], TS_RANGE[1])
            t_set = cloud.TaskSet(t_id, size, "high,low")
            t_set.generation(edge_sys, ub, uc)
            t_id += 1
            t_sets.append(t_set)

    for count in range(REPEAT_R):
        ub_ratio = random.uniform(UPPER_R[0], UPPER_R[1])
        uc_ratio = random.uniform(UPPER_R[0], UPPER_R[1])
        ub = ub_ratio * np.sum(edge_sys.ap_cap) / np.min(edge_sys.ap_cap)
        uc = uc_ratio * np.sum(edge_sys.s_cap) / np.min(edge_sys.s_cap)
        for count_s in range(REPEAT_SIZE):
            size = random.randint(TS_RANGE[0], TS_RANGE[1])
            t_set = cloud.TaskSet(t_id, size, "high,high")
            t_set.generation(edge_sys, ub, uc)
            t_id += 1
            t_sets.append(t_set)

    t_id_end = t_id
    utils.write(t_sets, "data/dataset_"+str(t_id_start)+"_"+str(t_id_end))


def run(cs: cloud.Cloud, tasksets, ts_id, al_id):
    ts: cloud.TaskSet
    alpha = ALPHA[al_id]
    solution_all = {}
    for ts in [tasksets[ts_id]]:
        print("=== Taskset ", ts.id, " starts ===")
        # ts: cloud.TaskSet = tasksets[-1]
        solution_ts = {"task_id": ts.id, "alpha": alpha, "base": BASE}

        # ====== Calculate GAA Solution ====== #
        sol_opt, sol_gaa, gma_time = ap.approximation(cs, ts, alpha, BASE, ts.id)
        if sol_opt == -1:
            continue
        solution_ts["opt"] = sol_opt
        sol_gaa["time"] = gma_time
        solution_ts["gaa"] = sol_gaa
        # print("GAA total spends %s seconds" % (gaa_end - gaa_start))
        # print(len(sol_gaa["mapping"]), " tasks are mapped by GAA!")

        # ====== Calculate ZSG Solution ====== #
        zsg_start = time.time()
        sol_zsg = gd.zero_slack_greedy(cs, ts, alpha)
        zsg_end = time.time()
        sol_zsg["time"] = zsg_end - zsg_start
        solution_ts["zsg"] = sol_zsg
        # print("ZSG total spends %s seconds" % (zsg_end - zsg_start))
        # print(len(sol_zsg["mapping"]), " tasks are mapped by ZSG!")

        # ====== Calculate ILP Solution ====== #
        ldm_time = gma_time
        sol_ldm = ldm.solve_ldm(cs, ts, alpha, ldm_time, ts.id)
        if sol_ldm == -1:
            continue
        solution_ts["ldm"] = sol_ldm

    utils.write(solution_all, "data/solution_alpha_" + str(alpha))


if __name__ == "__main__":
    # ====== Create Taskset and Cloud System ====== #
    random.seed(20230723)
    model_generation()
    create_tasksets2()

    # ====== Read Data from local file ====== #
    # cloud_sys: cloud.Cloud = utils.read("data/cloud_system.pickle")
    # tss = utils.read("data/dataset_0_3600.pickle")
    # print("file read done!")
    # index = 100
    # alpha_id = 0
    # run(cloud_sys, tss, index, alpha_id)
