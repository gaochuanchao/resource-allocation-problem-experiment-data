#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 5/18/2023 6:32 PM
# @Author  : Gao Chuanchao
# @Email   : jerrygao53@gmail.com
# @File    : draw_perfm_ratio.py
# add libraries
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import cloud
import utils
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    # Read data
    dataset = utils.read("data/dataset_0_3600.pickle")
    # dataset1 = utils.read("data/dataset_0_3600.pickle")
    # dataset2 = utils.read("data/dataset_0_3600.pickle")

    solution0 = utils.read("data/solution_all_alpha_0.pickle")
    solution1 = utils.read("data/solution_all_alpha_1.pickle")
    solution2 = utils.read("data/solution_all_alpha_2.pickle")

    solution_zsg0 = utils.read("data/solution_zsg_all_alpha_0.pickle")
    solution_zsg1 = utils.read("data/solution_zsg_all_alpha_1.pickle")
    solution_zsg2 = utils.read("data/solution_zsg_all_alpha_2.pickle")

    analysis = {"id": [], "Performance Ratio": [], "alg.": [], "Resource Utilization Level": []}

    for i in range(len(dataset)):
        ts: cloud.TaskSet = dataset[i]
        sol = solution0[ts.id]
        zsg_makeup = solution_zsg0[ts.id]

        sol_gaa = sol["gaa"]["energy"]
        # sol_zsg = sol["zsg"]["energy"]
        sol_zsg = zsg_makeup["zsg"]["energy"]
        sol_ldp = sol["ldm"]
        sol_opt = sol["opt"]

        analysis["id"].append(ts.id)
        analysis["Performance Ratio"].append(sol_gaa / sol_opt)
        analysis["alg."].append("GMA")

        analysis["id"].append(ts.id)
        analysis["Performance Ratio"].append(sol_zsg / sol_opt)
        analysis["alg."].append("ZSG")

        analysis["id"].append(ts.id)
        analysis["Performance Ratio"].append(sol_ldp / sol_opt)
        analysis["alg."].append("LDM")

        if ts.prop == "low,low":
            analysis["Resource Utilization Level"].append("[LR, LR]")
            analysis["Resource Utilization Level"].append("[LR, LR]")
            analysis["Resource Utilization Level"].append("[LR, LR]")
        elif ts.prop == "low,high":
            analysis["Resource Utilization Level"].append("[LR, HR]")
            analysis["Resource Utilization Level"].append("[LR, HR]")
            analysis["Resource Utilization Level"].append("[LR, HR]")
        elif ts.prop == "high,low":
            analysis["Resource Utilization Level"].append("[HR, LR]")
            analysis["Resource Utilization Level"].append("[HR, LR]")
            analysis["Resource Utilization Level"].append("[HR, LR]")
        else:
            analysis["Resource Utilization Level"].append("[HR, HR]")
            analysis["Resource Utilization Level"].append("[HR, HR]")
            analysis["Resource Utilization Level"].append("[HR, HR]")

    frame_data = pd.DataFrame.from_dict(analysis)
    # start draw graph
    rc('font', weight='bold')
    sns.set_style("whitegrid")
    ax = sns.boxplot(x="Resource Utilization Level", y="Performance Ratio",
                     order=["[LR, LR]", "[LR, HR]", "[HR, LR]", "[HR, HR]"], hue='alg.', data=frame_data,
                     palette="pastel", showmeans=True, showfliers=False,
            meanprops={'marker': 's', "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 4})
    # ax = sns.boxplot(x="Resource Usage Level", y="Performance Ratio",
    #                  order=["[LR, LR]", "[LR, HR]", "[HR, LR]", "[HR, HR]"], hue='alg.', data=frame_data,
    #                  palette="pastel", width=0.8, showmeans=True,
    #                meanprops={'marker': 's', "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 4})
    ax.set_xlabel("Resource Utilization Level $<r_b, r_c>$", fontsize=15, weight="bold")
    ax.set_ylabel("Performance Ratio $R$", fontsize=15, weight="bold")
    # ax.grid(linestyle="--", linewidth=0.5, color='.15', zorder=-10)
    ax.set_ylim(0.2, 1.01)
    plt.yticks(np.arange(0.2, 1.01, 0.1))
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    # plt.rcParams["figure.figsize"] = (7, 3)
    # ax.legend(loc="lower right", fontsize=12, ncol=3)
    ax.legend(loc="best", fontsize=12, ncol=3)
    plt.show()

    # top=0.98,
    # bottom=0.135,
    # left=0.09,
    # right=0.993,

