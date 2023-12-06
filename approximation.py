#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 5/3/2023 4:32 PM
# @Author  : Gao Chuanchao
# @Email   : jerrygao53@gmail.com
# @File    : approximation.py
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

getcontext().prec = 7
sys.setrecursionlimit(3000)


class BiGraph:
    """
    Define a bipartite graph.
        t_nodes: task nodes, {task_id}
        u_nodes: ap/server nodes, {(ap/server_id, r)}
        edges: edge set of the bipartite graph, {(task_id, (ap/server_id, r))}
        n_vals: number of nodes associated with each ap/server
        edge_resource: resource associated with each edge
    """

    def __init__(self):
        self.t_nodes = set()
        self.u_nodes = set()
        self.edges = set()
        self.n_vals = {}
        self.edge_resource = {}


class TriGraph:
    """
    Define a tripartite graph.
        nodes: including task node {task_id}, AP node {(ap_id, r, "ap")}, and server node {(server_id, r, "server")}
        edges: edges of this tripartite graph, {(task_id, (ap_id, r, "ap"), (server_id, s, "server"))}
        edge_icd_node: collection of edges incident to every node
        edges_comm_res: communication resource allocation for each edge
        edges_comp_res: computation resource allocation for each edge
        weight: weight associated with each edge, default = 1
    """

    def __init__(self):
        self.nodes = set()
        self.edges = set()
        self.edge_icd_node = dict()
        self.edges_comm_res = dict()
        self.edges_comp_res = dict()
        self.weight = dict()


def resource_discretization(cs: cloud.Cloud, alpha, base):
    # discretize communication resource allocation
    # minimum bandwidth 0.1 MHz
    resource_AP = []
    for j in range(cs.ap_count):
        pi_j = math.ceil(math.log(cs.ap_cap[j] * alpha * 10, base))
        B_j = []
        for m in range(pi_j):
            B_j.append(round(math.pow(base, m) / 10, 6))
        B_j.append(cs.ap_cap[j] * alpha)
        resource_AP.append(B_j)

    # discretize computation resource allocation
    resource_Server = []
    for k in range(cs.s_count):
        lambda_k = math.ceil(math.log(cs.s_cap[k] * alpha, base))
        C_k = []
        for n in range(lambda_k):
            C_k.append(round(math.pow(base, n), 6))
        C_k.append(cs.s_cap[k] * alpha)
        resource_Server.append(C_k)

    return resource_AP, resource_Server


def calc_saved_energy(size, b_ij, offload_t):
    log_term = size / (offload_t * b_ij)
    if log_term > 10:
        return -1
    power = (math.pow(2, log_term) - 1) * NOISE / CHANNEL_GAIN
    energy = power * offload_t

    if power > MAX_OFF_POWER:     # cap transmission power at 0.1W
        return -1
    else:
        return energy


def find_feasible_comb(ts: cloud.TaskSet, cs: cloud.Cloud, B_AP, C_Server, base):
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


def solve_optimal(comb_dict: dict, cs: cloud.Cloud, ts: cloud.TaskSet, B_AP: list, C_Server: list, base, ts_id):
    # define environments and LP models
    env = gp.Env()
    opt = gp.Model(name="OPT", env=env)
    opt.Params.LogToConsole = 0  # disable console logging
    opt.Params.MIPFocus = 1  # focus more on good quality feasible solutions
    opt.Params.TimeLimit = 600  # set a time limit of 10 minutes for the solver
    opt.Params.Threads = 32  # Number of parallel threads to use

    # define LDP decision variables z(i,j,m,k,n)
    z = opt.addVars(comb_dict["all"], ub=1.0, vtype=GRB.CONTINUOUS, name="z")

    # add constraints
    for i in range(ts.t_count):  # OPT-a
        if len(comb_dict["task"][i]) > 0:
            pair = [(1, z[comb]) for comb in comb_dict["task"][i]]
            opt.addConstr(gp.LinExpr(pair) <= 1, name="opt-a")

    for j in range(cs.ap_count):  # OPT-b
        if len(comb_dict["ap"][j]) > 0:
            pair = [(B_AP[j][comb[2]], z[comb]) for comb in comb_dict["ap"][j]]
            opt.addConstr(gp.LinExpr(pair) <= base * cs.ap_cap[j], name="opt-b")

    for k in range(cs.s_count):  # OPT-c
        if len(comb_dict["server"][k]) > 0:
            pair = [(C_Server[k][comb[4]], z[comb]) for comb in comb_dict["server"][k]]
            opt.addConstr(gp.LinExpr(pair) <= base * cs.s_cap[k], name="opt-c")

    # add objective function
    obj_pair = [(comb_dict["energy"][comb], z[comb]) for comb in comb_dict["all"]]
    opt.setObjective(gp.LinExpr(obj_pair), GRB.MAXIMIZE)

    # start to solve the LP
    opt.optimize()
    print("Taskset", ts_id, ": OPT is solved!")

    # record LP result
    # total_energy = 0
    # for comb in comb_dict["all"]:
    #     value = float(z[comb].X)
    #     if value > 0:
    #         total_energy += value * comb_dict["energy"][comb]
    if opt.SolCount >= 1:
        total_energy = opt.getObjective().getValue()
        # free all resources associated with this lp model and environment
        opt.dispose()
        env.dispose()
        return total_energy
    else:   # if no feasible solution, return -1
        msg = "Taskset " + str(ts_id) + " fails at function solve_optimal()\n"
        utils.log(msg, "exp1_data/log.txt")
        opt.dispose()
        env.dispose()
        return -1


def solve_LDP(comb_dict: dict, cs: cloud.Cloud, ts: cloud.TaskSet, B_AP: list, C_Server: list, alpha, ts_id):
    # define environments and LP models
    env = gp.Env()
    ldp = gp.Model(name="LDP", env=env)
    ldp.Params.LogToConsole = 0  # disable console logging
    ldp.Params.MIPFocus = 1  # focus more on good quality feasible solutions
    ldp.Params.TimeLimit = 600  # set a time limit of 10 minutes for the solver
    ldp.Params.Threads = 32  # Number of parallel threads to use

    # define LDP decision variables z(i,j,m,k,n)
    z = ldp.addVars(comb_dict["all"], ub=1.0, vtype=GRB.CONTINUOUS, name="z")

    # add constraints
    for i in range(ts.t_count):  # LDP-a
        if len(comb_dict["task"][i]) > 0:
            pair = [(1, z[comb]) for comb in comb_dict["task"][i]]
            ldp.addConstr(gp.LinExpr(pair) <= 1, name="ldp-a")

    for j in range(cs.ap_count):  # LDP-b
        if len(comb_dict["ap"][j]) > 0:
            pair = [(B_AP[j][comb[2]], z[comb]) for comb in comb_dict["ap"][j]]
            ldp.addConstr(gp.LinExpr(pair) <= (1 - alpha) * cs.ap_cap[j], name="ldp-b")

    for k in range(cs.s_count):  # LDP-c
        if len(comb_dict["server"][k]) > 0:
            pair = [(C_Server[k][comb[4]], z[comb]) for comb in comb_dict["server"][k]]
            ldp.addConstr(gp.LinExpr(pair) <= (1 - alpha) * cs.s_cap[k], name="ldp-c")

    # add objective function
    obj_pair = [(comb_dict["energy"][comb], z[comb]) for comb in comb_dict["all"]]
    ldp.setObjective(gp.LinExpr(obj_pair), GRB.MAXIMIZE)

    # start to solve the LP
    ldp.optimize()
    print("Taskset", ts_id, ": LDP is solved!")

    # record LP result
    if ldp.SolCount >= 1:
        positive_comb = {}
        for comb in comb_dict["all"]:
            value = Decimal(z[comb].X)
            if value > 0:
                positive_comb[comb] = value
        # obj_LDP = ldp.getObjective().getValue()
        # print("LDP Objective Value is ", obj_LDP)

        # free all resources associated with this lp model and environment
        ldp.dispose()
        env.dispose()

        return positive_comb
    else:   # if no feasible solution, return -1
        msg = "Taskset " + str(ts_id) + " fails at function solve_LDP()\n"
        utils.log(msg, "exp1_data/log.txt")
        ldp.dispose()
        env.dispose()
        return -1


def ldp_result_analysis(ldp_sol: dict, cs: cloud.Cloud):
    # obtain x_ijm, y_ikn
    sol_AP = [{} for j in range(cs.ap_count)]
    sol_Server = [{} for k in range(cs.s_count)]
    sol_Task = {}

    for comb in ldp_sol.keys():  # comb: (i,j,m,k,n)
        # obtain x_ijm
        x_comb = (comb[0], comb[1], comb[2])
        value = sol_AP[comb[1]].get(x_comb, 0)
        sol_AP[comb[1]][x_comb] = value + ldp_sol[comb]

        # obtain y_ikn
        y_comb = (comb[0], comb[3], comb[4])
        value = sol_Server[comb[3]].get(y_comb, 0)
        sol_Server[comb[3]][y_comb] = value + ldp_sol[comb]

        # obtain z_i
        value = sol_Task.get(comb[0], 0)
        sol_Task[comb[0]] = value + ldp_sol[comb]

    return {"task": sol_Task, "ap": sol_AP, "server": sol_Server}


def find_index_lr(posi_j, X_j):
    """
    Find the minimum index i such that sum(X_j[:i]) >= r
    :param posi_j: {(i,j,m): x_ijm}
    :param X_j: the ranked combination {(i,j,m)}
    :return: list of index, [l_r1, l_r2, ...]
    """
    x_sum = Decimal(0)
    threshold = Decimal(1)
    lr_index = []
    sum_up_to_lr = []
    for s in range(len(X_j)):
        x_sum += posi_j[X_j[s]]
        # x_sum = round(x_sum, 12)    # use rounding to avoid floating value precision issue
        if x_sum >= threshold:
            lr_index.append(s)
            sum_up_to_lr.append(x_sum)
            threshold += 1

    return lr_index, sum_up_to_lr


def assign_edges(e, graph: BiGraph, resource):
    """
    add edges to graph and assign corresponding resource allocation
    :param e: edge to be added
    :param graph: the bipartite graph
    :param resource: resource allocation to edge e
    """
    if e not in graph.edges:
        graph.edges.add(e)
        graph.edge_resource[e] = resource


def bipartite_graph_construct(sol_data: dict, unit_count, unit_t: str, res_U: list, ts_id):
    """
    Construct Bi-partite Graphs
    :param sol_data: {"task": sol_Task, "ap": sol_AP, "server": sol_Server}
    :param unit_count: number of AP or server
    :param unit_t: unit type, "ap" or "server"
    :param res_U: discretized resource, B_AP or C_Server
    :return:
    """
    graph = BiGraph()
    sol_unit = sol_data[unit_t]  # solution of AP or server with positive value
    # record task nodes
    graph.t_nodes = set(sol_data["task"].keys())
    # record nodes associated with each unit
    graph.n_vals = {}
    for c in range(unit_count):
        sum_v = Decimal(0)
        values = list(sol_unit[c].values())   # i.e., sol_unit[c] = {(i,j,m): x_ijm}
        for v in values:
            sum_v += v
        graph.n_vals[c] = math.ceil(sum_v)
        for r in range(graph.n_vals[c]):
            graph.u_nodes.add((c, r))

    # construct edges
    special_comb = set()  # include those x_ijm or y_ikn that result in two edges
    for c in range(unit_count):
        X_j = list(sol_unit[c].keys())  # i.e., sol_unit[c] = {(i,j,m): x_ijm}
        # sort tuple based on their resource allocation (m-value)
        X_j.sort(key=lambda combi: combi[2], reverse=True)
        if graph.n_vals[c] == 1:
            for comb in X_j:  # comb = (i,j,m)
                assign_edges((comb[0], (c, 0)), graph, res_U[c][comb[2]])  # res_U[c] = B_AP[j] or C_Server[k]
        elif graph.n_vals[c] > 1:
            try:
                list_lr, sum_to_lr = find_index_lr(sol_unit[c], X_j)  # obtain l_r, sum_to_lr
                for r in range(graph.n_vals[c] - 1):  # r = 0, 1, ..., graph.n_vals[c] - 2
                    start_s = 0  # set start index
                    if r >= 1:  # when r = 0, list also start from 0
                        start_s = list_lr[r - 1] + 1

                    for s in range(start_s, list_lr[r] + 1):  # s = start_i, ..., list_lr[r]
                        comb = X_j[s]  # comb = (i,j,m)
                        assign_edges((comb[0], (c, r)), graph, res_U[c][comb[2]])
                    if sum_to_lr[r] > (r + 1):  # when the sum is strictly larger than r+1
                        comb = X_j[list_lr[r]]
                        assign_edges((comb[0], (c, r + 1)), graph, res_U[c][comb[2]])
                        special_comb.add(comb)  # record this comb if it results two edges

                # handle remaining tasks, s = list_lr[graph.n_vals[c]-2]+1, ..., |X_j|
                start_s = list_lr[graph.n_vals[c] - 2] + 1  # list_lr[r-1] when r=n_vals[j]-1
                for s in range(start_s, len(X_j)):
                    comb = X_j[s]
                    assign_edges((comb[0], (c, graph.n_vals[c] - 1)), graph, res_U[c][comb[2]])
            except IndexError:
                msg = "Taskset " + str(ts_id) + " fails at function bipartite_graph_construct() [IndexError!]\n"
                utils.log(msg, "exp1_data/log.txt")
                return -1, -1

    return graph, special_comb


def tripartite_graph_construction(graph_x: BiGraph, graph_y: BiGraph, posi_comb,feasible_comb,
                                  B_AP, C_Server, spec_ijm, spec_ikn):
    triGraph = TriGraph()
    # add nodes, need differentiate ap nodes and server nodes
    triGraph.nodes.update(graph_x.t_nodes)
    for node_ap in graph_x.u_nodes:
        triGraph.nodes.add((node_ap[0], node_ap[1], "ap"))
    for node_s in graph_y.u_nodes:
        triGraph.nodes.add((node_s[0], node_s[1], "server"))

    # add edges
    for comb in posi_comb:  # comb = (i, j, m, k, n)
        energy = feasible_comb["energy"][comb]
        E_ijm = set()
        for r in range(graph_x.n_vals[comb[1]] - 1, -1, -1):
            e = (comb[0], (comb[1], r))
            if (e in graph_x.edges) and (graph_x.edge_resource[e] >= B_AP[comb[1]][comb[2]]):
                E_ijm.add(e)
            e2 = (comb[0], (comb[1], r - 1))  # extra possible edge
            if ((comb[0], comb[1], comb[2]) in spec_ijm) and (e2 in graph_x.edges):
                E_ijm.add(e2)

        E_ikn = set()
        for r in range(graph_y.n_vals[comb[3]] - 1, -1, -1):
            e = (comb[0], (comb[3], r))
            if (e in graph_y.edges) and (graph_y.edge_resource[e] >= C_Server[comb[3]][comb[4]]):
                E_ikn.add(e)
            e2 = (comb[0], (comb[3], r - 1))  # extra possible edge
            if ((comb[0], comb[3], comb[4]) in spec_ikn) and (e2 in graph_y.edges):
                E_ikn.add(e2)

        for e_x in E_ijm:  # e_x = (i, (j, r))
            for e_y in E_ikn:  # e_y = (i, (k, s))
                w1 = (e_x[1][0], e_x[1][1], "ap")
                w2 = (e_y[1][0], e_y[1][1], "server")
                if (comb[0], w1, w2) in triGraph.edges:
                    if energy > triGraph.weight[(comb[0], w1, w2)]:
                        triGraph.weight[(comb[0], w1, w2)] = energy
                else:
                    # update edge and wights
                    triGraph.edges.add((comb[0], w1, w2))
                    triGraph.weight[(comb[0], w1, w2)] = energy
                    triGraph.edges_comm_res[(comb[0], w1, w2)] = graph_x.edge_resource[e_x]
                    triGraph.edges_comp_res[(comb[0], w1, w2)] = graph_y.edge_resource[e_y]

                    # update edges incident to node i
                    e_i = triGraph.edge_icd_node.setdefault(comb[0], set())
                    e_i.add((comb[0], w1, w2))
                    # update edges incident to node (j,r,"ap")
                    e_jr = triGraph.edge_icd_node.setdefault(w1, set())
                    e_jr.add((comb[0], w1, w2))
                    # update edges incident to node (k,s,"server")
                    e_ks = triGraph.edge_icd_node.setdefault(w2, set())
                    e_ks.add((comb[0], w1, w2))

    return triGraph


def solve_3DM(triGraph: TriGraph, ts_id):
    # define environments and LP models 3DM (Tripartite Graph Matching)
    env = gp.Env()
    tgm = gp.Model(name="TGM", env=env)
    tgm.Params.LogToConsole = 0  # disable console logging
    tgm.Params.TimeLimit = 600  # set a time limit of 10 minutes for the solver
    tgm.Params.Threads = 32  # Number of parallel threads to use

    # define lp1 decision variables z_e
    edge_list = list(triGraph.edges)
    z = tgm.addVars(edge_list, ub=1.0, vtype=GRB.CONTINUOUS, name="z")

    # add constraints
    for node in triGraph.nodes:
        edges = triGraph.edge_icd_node.setdefault(node, set())
        if len(edges) > 0:
            pair = [(1, z[e]) for e in edges]
            tgm.addConstr(gp.LinExpr(pair) <= 1, name="tgm-a")

    # add objective function
    obj_pair = [(triGraph.weight[e], z[e]) for e in edge_list]
    tgm.setObjective(gp.LinExpr(obj_pair), GRB.MAXIMIZE)

    # start to solve the LP
    tgm.optimize()

    print("Taskset", ts_id, ": 3DM is solved!")

    # record LP result
    # obj_value = tgm.getObjective().getValue()
    frac_matching = dict()
    for edge in edge_list:
        value = Decimal(z[edge].X)
        if value > 0:
            frac_matching[edge] = value
    # free all resources associated with this lp model and environment
    tgm.dispose()
    env.dispose()

    return frac_matching


def rounding(tg: TriGraph, fm: dict):
    """
    rounding method for converting a fractional matching to an integral matching
    :param tg: a tripartite graph
    :param fm: a fractional matching of the provided graph
    :return: an integral matching of the provided graph
    """
    # record edges with z(e) > 0
    edges = set(fm.keys())
    # record edge intersection
    isc_edges = dict()
    for e in edges:
        full_isc_set = set()
        full_isc_set.update(tg.edge_icd_node[e[0]], tg.edge_icd_node[e[1]], tg.edge_icd_node[e[2]])
        # discard edges with z(e) = 0
        isc_edges[e] = edges.intersection(full_isc_set)

    ordered_edges = []
    size = len(edges)
    for i in range(size):
        e = find_edge(edges, isc_edges, ordered_edges, fm, 2)
        ordered_edges.append(e)
        edges.discard(e)
    weight = {e: tg.weight[e] for e in ordered_edges}

    # avoid RecursionError, reset limit after code execution
    old_limit = sys.getrecursionlimit()
    if old_limit <= len(ordered_edges):
        sys.setrecursionlimit(len(ordered_edges) + 1000)
        matching = local_ratio(ordered_edges, weight, isc_edges)
        # sys.setrecursionlimit(old_limit)
        return matching
    else:
        return local_ratio(ordered_edges, weight, isc_edges)


def find_edge(edges: set, isc_edges: dict, od_edges: list, fm: dict, bound: int):
    """
    Find an edge that the sum of fractions of edges intersect with it no larger than the bound
    :param edges: set of edges with fraction > 0
    :param isc_edges: edge intersections
    :param od_edges: a list of ordered edges
    :param fm: a fractional matching of tg
    :param bound: the bound for edge fraction sum
    :return: an edge
    """
    for e in edges:
        isc_e = {i for i in isc_edges[e]}  # avoid manipulating original edge set
        # remove edges that are already added to the ordered edge list
        for edge in od_edges:
            isc_e.discard(edge)
        # check the sum of remaining intersection edges
        total = sum([fm[i] for i in isc_e])
        if total <= bound:
            return e


def local_ratio(od_edges: list, weight: dict, isc_edges: dict):
    """
    local-ratio method
    :param od_edges: ordered edges
    :param weight: weight of all edges in od_edges
    :param isc_edges: edge intersections
    :return: integral matching
    """
    # remove all edges with non-positive weights
    edges = [e for e in od_edges if weight[e] > 0]
    w = {e: weight[e] for e in edges}
    if len(edges) == 0:
        return set()
    # find the edge with the smallest index
    es = edges[0]
    w_es = w[es]
    # set of edges intersect with edge
    isc_es = isc_edges[es]
    # split edge weight
    w1 = dict()
    w2 = dict()
    for e in edges:
        if e in isc_es:
            w1[e] = w_es
        else:
            w1[e] = 0
        w2[e] = w[e] - w1[e]
        if w2[e] > w[e]:
            raise Exception("Wrong value for w2[", e, "]! w[e] = ", w[e], ", w2[e] = ", w2[e])

    matching = local_ratio(edges, w2, isc_edges)
    for edge in matching:
        if edge in isc_es:
            return matching

    matching.add(es)
    return matching


def approximation(cs: cloud.Cloud, ts: cloud.TaskSet, alpha, base, ts_id):
    """
    Approximation algorithm
    :param cs: edge-computing system
    :param ts: tasks pending for allocation
    :param alpha: resource allocation bound
    :param base: base value for resource discretization
    :return: integral task mapping result
    """
    start_time = time.time()
    # Step 1: resource discretization and LDP formulation
    B_AP, C_Server = resource_discretization(cs, alpha, base)
    feasible_comb = find_feasible_comb(ts, cs, B_AP, C_Server, base)
    time2 = time.time()

    # find optimal upper bound
    opt_sol = solve_optimal(feasible_comb, cs, ts, B_AP, C_Server, base, ts_id)
    if opt_sol == -1:
        return -1, -1, -1

    # ========= start solving GMA ===============
    time3 = time.time()
    ldp_sol = solve_LDP(feasible_comb, cs, ts, B_AP, C_Server, alpha, ts_id)  # includes all positive combs
    if ldp_sol == -1:
        return -1, -1, -1

    # Step 2: construct bipartite graphs
    ldp_summary = ldp_result_analysis(ldp_sol, cs)
    bi_graph_x, spec_ijm = bipartite_graph_construct(ldp_summary, cs.ap_count, "ap", B_AP, ts_id)
    if bi_graph_x == -1:
        return -1, -1, -1
    bi_graph_y, spec_ikn = bipartite_graph_construct(ldp_summary, cs.s_count, "server", C_Server, ts_id)
    if bi_graph_y == -1:
        return -1, -1, -1

    # Step 3: construct tripartite graph
    tri_graph = tripartite_graph_construction(bi_graph_x, bi_graph_y, ldp_sol, feasible_comb,
                                              B_AP, C_Server, spec_ijm, spec_ikn)
    frac_matching = solve_3DM(tri_graph, ts_id)

    # Step 4: obtain an integral matching
    matching = rounding(tri_graph, frac_matching)

    mapping = []
    comm_res = dict()
    comp_res = dict()
    total_energy = 0
    for edge in matching:  # edge = (i, (j, r, "ap"), (k, s, "server"))
        comb = (edge[0], edge[1][0], edge[2][0])
        mapping.append(comb)
        comm_res[comb] = tri_graph.edges_comm_res[edge]
        comp_res[comb] = tri_graph.edges_comp_res[edge]
        total_energy += tri_graph.weight[edge]
    end_time = time.time()

    sub_time = end_time - time3
    running_time = (time2 - start_time) + sub_time
    # print("remaining part spends %s seconds" % (end_time - time3))
    # print("GAA total spends %s seconds" % (end_time - start_time))
    gma_sol = {"mapping": mapping, "energy": total_energy, "comm_res": comm_res,
               "comp_res": comp_res, "sub_time": sub_time}

    return opt_sol, gma_sol, running_time


def optimal(cs: cloud.Cloud, ts: cloud.TaskSet, alpha, base, ts_id):
    B_AP, C_Server = resource_discretization(cs, alpha, base)
    feasible_comb = find_feasible_comb(ts, cs, B_AP, C_Server, base)
    opt_sol = solve_optimal(feasible_comb, cs, ts, B_AP, C_Server, base, ts_id)

    return opt_sol
