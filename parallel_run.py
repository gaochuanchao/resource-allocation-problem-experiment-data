#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 5/17/2023 11:03 AM
# @Author  : Gao Chuanchao
# @Email   : jerrygao53@gmail.com
# @File    : parallel_run.py
import cloud
import utils
from config import *
import approximation as ap
import time
import greedy as gd
import ldm
from multiprocessing import Process


def run(start_id, end_id, alpha):
    # ====== Read Data from local file ====== #
    cs: cloud.Cloud = utils.read("data/cloud_system.pickle")
    tasksets = utils.read("data/dataset_0_3600.pickle")

    solution_all = {}
    ts: cloud.TaskSet
    for ts in tasksets[start_id:end_id]:
        print("=== Taskset ", ts.id, " starts ===")
        # ts: cloud.TaskSet = tasksets[-1]
        solution_ts = {"task_id": ts.id, "alpha": alpha, "base": BASE}

        # ====== Calculate OPT & GAA Solution ====== #
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

        solution_all[ts.id] = solution_ts

    utils.write(solution_all, "data/subresult/solution_"+str(start_id)+"_"+str(end_id)+"_alpha_0")


def summarize_result():
    solutions = {}
    for i in range(24):
        start_id = i*100
        end_id = (i+1)*100
        sol = utils.read("data/subresult/solution_"+str(start_id)+"_"+str(end_id)+"_alpha_0.pickle")
        solutions.update(sol)

    utils.write(solutions, "data/solution_all_alpha_0")


if __name__ == "__main__":
    # ====== Creating Multi-processes ====== #
    alpha_v = ALPHA[0]
    processes = []
    for i in range(24):
        s_idx = i*100
        e_idx = (i+1)*100
        p = Process(target=run, args=(s_idx, e_idx, alpha_v,))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("All process ended!")

    # ====== combine solutions ====== #
    summarize_result()
