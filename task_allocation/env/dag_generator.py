import random, math, argparse
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
random.seed(50)
np.random.seed(50)


set_dag_size = [20, 30, 40, 50, 60, 70, 80, 90]  # random number of DAG  nodes
set_max_out = [1, 2, 3, 4, 5]  # max out_degree of one node
set_alpha = [0.5, 1.0, 1.5]  # DAG shape
set_beta = [0.0, 0.5, 1.0, 2.0]  # DAG regularity


def DAGs_generate(n=10, max_out=2, alpha=1, beta=1.0):

    length = math.floor(math.sqrt(n) / alpha)
    mean_value = n / length
    random_num = np.random.normal(loc=mean_value, scale=beta, size=(length, 1))
    ###############################################division#############################################
    position = {"Start": (0, 4), "Exit": (10, 4)}
    generate_num = 0
    dag_num = 1
    dag_list = []
    for i in range(len(random_num)):
        dag_list.append([])
        for j in range(math.ceil(random_num[i])):
            dag_list[i].append(j)
        generate_num += len(dag_list[i])

    if generate_num != n:
        if generate_num < n:
            for i in range(n - generate_num):
                index = random.randrange(0, length, 1)
                dag_list[index].append(len(dag_list[index]))
        if generate_num > n:
            i = 0
            while i < generate_num - n:
                index = random.randrange(0, length, 1)
                if len(dag_list[index]) <= 1:
                    continue
                else:
                    del dag_list[index][-1]
                    i += 1

    dag_list_update = []
    pos = 1
    max_pos = 0
    for i in range(length):
        dag_list_update.append(list(range(dag_num, dag_num + len(dag_list[i]))))
        dag_num += len(dag_list_update[i])
        pos = 1
        for j in dag_list_update[i]:
            position[j] = (3 * (i + 1), pos)
            pos += 5
        max_pos = pos if pos > max_pos else max_pos
        position["Start"] = (0, max_pos / 2)
        position["Exit"] = (3 * (length + 1), max_pos / 2)

    ############################################link#####################################################
    into_degree = [0] * n
    out_degree = [0] * n
    edges = []
    pred = 0

    for i in range(length - 1):
        sample_list = list(range(len(dag_list_update[i + 1])))
        for j in range(len(dag_list_update[i])):
            od = random.randrange(1, max_out + 1, 1)
            od = len(dag_list_update[i + 1]) if len(dag_list_update[i + 1]) < od else od
            bridge = random.sample(sample_list, od)
            for k in bridge:
                weight = random.randint(2000000, 5000000)
                edges.append(
                    (
                        dag_list_update[i][j],
                        dag_list_update[i + 1][k],
                        weight,
                    )
                )
                into_degree[pred + len(dag_list_update[i]) + k] += 1
                out_degree[pred + j] += 1
        pred += len(dag_list_update[i])

    ######################################create start node and exit node################################
    for node, id in enumerate(into_degree):  # 给所有没有入边的节点添加入口节点作父亲
        if id == 0:
            edges.append(("Start", node + 1, 0))
            into_degree[node] += 1

    for node, od in enumerate(out_degree):  # 给所有没有出边的节点添加出口节点作儿子
        if od == 0:
            edges.append((node + 1, "Exit", 0))
            out_degree[node] += 1

    #############################################plot###################################################
    return edges, into_degree, out_degree, position


def plot_DAG(edges, postion):
    g1 = nx.DiGraph()
    g1.add_weighted_edges_from(edges)
    pos = postion
    nx.draw(g1, pos, with_labels=True, node_color="lightblue")
    nx.draw_networkx_edge_labels(
        g1,
        pos,
        edge_labels={(u, v): f"{d['weight']}" for u, v, d in g1.edges(data=True)},
        font_color="red",
    )
    plt.show()
    return plt.clf


def workflows_generator(
    n=20, max_out=2, alpha=1, beta=1.0, t_unit=10, resource_unit=100
):
    workflows = []
    for max_out in set_max_out:
        for alpha in set_alpha:
            for beta in set_beta:
                edges, in_degree, out_degree, position = DAGs_generate(
                    n, max_out, alpha, beta
                )
                path = "m,a,b: " + str(max_out) + "," + str(alpha) + "," + str(beta)
                runtimes = []
                stores = []
                for _ in range(n):
                    runtimes.append(random.randint(5000, 10000))
                    stores.append(random.uniform(0.1, 10.0))
                workflows.append((edges, runtimes, stores, path, position))

    return workflows
