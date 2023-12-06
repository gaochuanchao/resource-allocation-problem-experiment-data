#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 5/15/2023 7:19 PM
# @Author  : Gao Chuanchao
# @Email   : jerrygao53@gmail.com
# @File    : ilp.py
import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import cloud
import sys
import time
from decimal import *

import utils
from config import *


def resource_discretization(cs: cloud.Cloud, alpha):
    # discretize communication resource allocation
    resource_AP = []
    for j in range(cs.ap_count):
        pi_j = math.floor(cs.ap_cap[j] * alpha / UNIT_B)
        B_j = []
        for m in range(1, pi_j+1):
            B_j.append(UNIT_B * m)
        resource_AP.append(B_j)

    # discretize computation resource allocation
    resource_Server = []
    for k in range(cs.s_count):
        lambda_k = math.floor(cs.s_cap[k] * alpha / UNIT_C)
        C_k = []
        for n in range(1, lambda_k+1):
            C_k.append(UNIT_C * n)
        resource_Server.append(C_k)

    return resource_AP, resource_Server


def calc_saved_energy(size, b_ij, offload_t):
    log_term = size / (offload_t * b_ij)
    if log_term > 10:
        return -1
    power = (math.pow(2, log_term) - 1) * NOISE / CHANNEL_GAIN
    energy = power * offload_t

    if power > MAX_OFF_POWER:  # cap transmission power at 0.1W
        return -1
    else:
        return energy


def find_feasible_comb(ts: cloud.TaskSet, cs: cloud.Cloud, B_AP, C_Server):
    # find all feasible combinations
    comb_all = []
    comb_Task = {i: [] for i in range(ts.t_count)}
    comb_AP = {j: [] for j in range(cs.ap_count)}
    comb_Server = {k: [] for k in range(cs.s_count)}
    saved_energy = {}

    for i in range(ts.t_count):
        # obtain the access points that task i can be offloaded to
        task_links = ts.task_ap_link[i]
        ap_i = list(np.where(task_links == 1)[0])
        for j in ap_i:
            for k in range(cs.s_count):
                for m in range(len(B_AP[j])):
                    for n in range(len(C_Server[k])):
                        process_time = ts.cycle[i] / C_Server[k][n]
                        offload_time = ts.deadline[i] - (cs.delta[j, k] + process_time)
                        if offload_time > 0:
                            off_energy = calc_saved_energy(ts.size[i], B_AP[j][m], offload_time)
                            if off_energy > 0:
                                saved_e = ts.local_energy[i] - off_energy
                                if saved_e > 0:
                                    comb_all.append((i, j, m, k, n))
                                    comb_Task[i].append((i, j, m, k, n))
                                    comb_AP[j].append((i, j, m, k, n))
                                    comb_Server[k].append((i, j, m, k, n))
                                    saved_energy[(i, j, m, k, n)] = saved_e

    return {"all": comb_all, "task": comb_Task, "ap": comb_AP, "server": comb_Server, "energy": saved_energy}


def solve_optimal(comb_dict: dict, cs: cloud.Cloud, ts: cloud.TaskSet, B_AP: list, C_Server: list, time_limit, ts_id):
    # define environments and LP models
    env = gp.Env()
    opt = gp.Model(name="ILP", env=env)
    opt.Params.LogToConsole = 0  # disable console logging
    opt.Params.MIPFocus = 1  # focus more on good quality feasible solutions
    opt.Params.TimeLimit = time_limit  # set a time limit of 10 minutes for the solver
    opt.Params.Threads = 32  # Number of parallel threads to use

    # define LDP decision variables z(i,j,m,k,n)
    z = opt.addVars(comb_dict["all"], vtype=GRB.BINARY, name="z")

    # add constraints
    for i in range(ts.t_count):  # OPT-a
        if len(comb_dict["task"][i]) > 0:
            pair = [(1, z[comb]) for comb in comb_dict["task"][i]]
            opt.addConstr(gp.LinExpr(pair) <= 1, name="opt-a")

    for j in range(cs.ap_count):  # OPT-b
        if len(comb_dict["ap"][j]) > 0:
            pair = [(B_AP[j][comb[2]], z[comb]) for comb in comb_dict["ap"][j]]
            opt.addConstr(gp.LinExpr(pair) <= cs.ap_cap[j], name="opt-b")

    for k in range(cs.s_count):  # OPT-c
        if len(comb_dict["server"][k]) > 0:
            pair = [(C_Server[k][comb[4]], z[comb]) for comb in comb_dict["server"][k]]
            opt.addConstr(gp.LinExpr(pair) <= cs.s_cap[k], name="opt-c")

    # add objective function
    obj_pair = [(comb_dict["energy"][comb], z[comb]) for comb in comb_dict["all"]]
    opt.setObjective(gp.LinExpr(obj_pair), GRB.MAXIMIZE)

    # start to solve the LP
    opt.optimize()
    print("Taskset ", ts_id, " ILP is solved!")

    # record LP result
    if opt.SolCount >= 1:
        total_energy = opt.getObjective().getValue()
        # free all resources associated with this lp model and environment
        opt.dispose()
        env.dispose()
        return total_energy
    else:   # if no feasible solution, return -1
        msg = "Taskset " + str(ts_id) + " fails at function solve_LDM()\n"
        utils.log(msg, "exp1_data/log.txt")
        opt.dispose()
        env.dispose()
        return -1


def solve_ldm(cs: cloud.Cloud, ts: cloud.TaskSet, alpha, time_limit, ts_id):
    time1 = time.time()
    B_AP, C_Server = resource_discretization(cs, alpha)
    feasible_comb = find_feasible_comb(ts, cs, B_AP, C_Server)
    time2 = time.time()
    remaining_time = max(time_limit - (time2 - time1), 60)
    ldm_sol = solve_optimal(feasible_comb, cs, ts, B_AP, C_Server, remaining_time, ts_id)

    return ldm_sol
