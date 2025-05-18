import enum
import re
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse as sp
import torch
import random
from collections import deque


class TaskStatus(enum.Enum):
    none = 0
    pool = 1
    # 2: add to ready queue to be scheduled
    ready = 2
    wait = 3
    # 4: The task is running
    run = 4
    done = 5


class Task:
    def __init__(self, num, length):
        self.id = "wf?-" + str(num)
        self.name = "wf?-" + str(num)
        self.num = num
        self.length = length
        self.uprank = 0
        self.rank_exe = 0
        self.rank_trans = 0
        self.status = TaskStatus.none

        self.depth = -1
        self.depth_len = -1
        self.input_size = 0
        self.output_size = 0
        self.succ = []
        self.pred = []
        self.input_files = []
        self.output_files = []

        # 任务进入工作流的准备时间
        self.ready_time = 0
        self.estimate_finish_time = 0
        self.trans_time = 0
        self.exe_time = 0
        self.vm_queue_time = 0
        self.start_time = 0
        self.finish_time = 0

        self.deadline = -1
        self.budget = -1

        self.workflow = None
        # 保存 VM 引用， 创建对象的弱引用
        self.vm = None
        # 将虚拟机作为键，时间和成本作为值，存储在任务的 vref_time_cost 属性中
        self.fast_run = 0
        self.vm_time_cost = {}  #  dictionary {v1:[2,0.5] v2: [23, 5] ......}

    def setWorkflow(self, wf):
        self.workflow = wf
        self.id = str(wf.id) + "-" + str(self.num)

    def isReadyToSch(self):
        for parent in self.pred:
            if parent.status != TaskStatus.done:
                return False
        return True

    def isAllChildrenDone(self):
        for child in self.succ:
            if child.status != TaskStatus.done:
                return False
        return True


class File:
    def __init__(self, name, size):
        self.name = str(name)
        self.size = float(size)


def setTaskDepth(task, d, l):
    if task.depth < d:
        task.depth = d
    if task.depth_len < l:
        task.depth_len = l

    for child_task in task.succ:
        setTaskDepth(child_task, task.depth + 1, task.depth_len + task.length)


def setTaskEntryAndExit(tasks):
    # Add an entry task and an exit task to the workflow
    roots = []
    lasts = []

    # 处理头结点和尾节点的依赖关系
    for task in tasks:
        task.depth = 0
        if len(task.pred) == 0:
            roots.append(task)
        elif len(task.succ) == 0:
            lasts.append(task)

    entry_task = Task(0, 0)
    exit_task = Task(-1, 0)
    for task in roots:
        task.pred.append(entry_task)
        entry_task.succ.append(task)
        for f in task.input_files:
            entry_task.output_files.append(f)
    for task in lasts:
        task.succ.append(exit_task)
        exit_task.pred.append(task)
        for f in task.output_files:
            exit_task.input_files.append(f)
    tasks.append(entry_task)
    tasks.append(exit_task)

    # Calculate each task's depth
    setTaskDepth(entry_task, 0, 0)

    # 计算任务的输入和输出文件的大小
    for task in tasks:
        for input_file in task.input_files:
            task.input_size += input_file.size
        for output_file in task.output_files:
            task.output_size += output_file.size

    return tasks


def find_critical_path(tasks):
    # 任务的属性初始化
    task_map = {task.num: task for task in tasks}
    in_degree = {task.num: len(task.pred) for task in tasks}

    # Step 1: 拓扑排序
    topo_order = []
    zero_in_degree = deque([task.num for task in tasks if in_degree[task.num] == 0])

    while zero_in_degree:
        current = zero_in_degree.popleft()
        topo_order.append(current)
        for succ in task_map[current].succ:
            in_degree[succ.num] -= 1
            if in_degree[succ.num] == 0:
                zero_in_degree.append(succ.num)

    # Step 2: 计算最早开始时间（EST）
    est = {task.num: 0 for task in tasks}
    for task_id in topo_order:
        task = task_map[task_id]
        for succ in task.succ:
            est[succ.num] = max(est[succ.num], est[task_id] + task.length)

    # Step 3: 计算最晚允许时间（LST）
    lst = {task.num: float("inf") for task in tasks}
    for task_id in reversed(topo_order):
        task = task_map[task_id]
        if not task.succ:  # 汇节点
            lst[task_id] = est[task_id]
        for succ in task.succ:
            lst[task_id] = min(lst[task_id], lst[succ.num] - task.length)

    # Step 4: 确定关键路径
    critical_path = []
    critical_length = 0
    for task_id in topo_order:
        if est[task_id] == lst[task_id]:
            critical_path.append(task_id)
            critical_length += task_map[task_id].length

    return critical_path, critical_length


def parse_xhn_tasks(workflow):
    tasks_name = workflow[0]
    adj_matrix = workflow[1]
    tasks_size = workflow[2]
    num_nodes = len(tasks_name)

    tasks = []
    files = []
    tasks = [Task(tasks_name[i], tasks_size[i]) for i in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                tasks[i].succ.append(tasks[j])
                tasks[j].pred.append(tasks[i])
                file_size = random.randint(2000000, 5000000)
                file = File(
                    "f" + str(tasks[i].name) + "-" + str(tasks[j].name), file_size
                )
                tasks[i].output_files.append(file)
                tasks[j].input_files.append(file)

    tasks = setTaskEntryAndExit(tasks)
    return tasks, files


def parse_generate_dag(workflow):
    edges = workflow[0]
    runtimes = workflow[1]
    tasks = []
    files = []
    for i, runtime in enumerate(runtimes):
        task = Task(i + 1, runtime)
        tasks.append(task)

    for edge in edges:
        if edge[0] == "Start" or edge[1] == "Exit":
            break
        parent = tasks[edge[0] - 1]
        child = tasks[edge[1] - 1]
        parent.succ.append(child)
        child.pred.append(parent)

        file_name = "f" + str(edge[0]) + "-" + str(edge[1])
        file_size = float(edge[2])
        file = File(file_name, file_size)
        parent.output_files.append(file)
        child.input_files.append(file)

    tasks = setTaskEntryAndExit(tasks)
    return tasks, files


def parseDAX(xmlfile):
    tasks = []
    files = []

    def getTask(num):
        for task in tasks:
            if task.num == num:
                return task

    def convertTaskRealIdToNum(id_str):
        return int(re.findall("\d+", id_str)[0]) + 1

    tree = ET.parse(xmlfile)
    root = tree.getroot()
    for node in root:
        if "job" in node.tag.lower():
            num = convertTaskRealIdToNum(node.attrib.get("id"))
            runtime = float(node.attrib.get("runtime")) * 100
            task = Task(num, runtime)
            tasks.append(task)

            # 被用来检查文件的大小、名称，以及它是输入文件还是输出文件。
            for file in node:
                if "uses" in file.tag.lower():
                    file_size = float(file.attrib.get("size"))
                    file_name = file.attrib.get("file")

                    if file.attrib.get("link") == "output":
                        file_already_exist = None
                        for file in files:
                            if file_name == file.name:
                                file_already_exist = file
                                task.output_files.append(file)
                        if not file_already_exist:
                            file_item = File(file_name, file_size)
                            files.append(file_item)
                            task.output_files.append(file_item)

                    elif file.attrib.get("link") == "input":
                        file_already_exist = None
                        for file in files:
                            if file_name == file.name:
                                file_already_exist = file
                                task.input_files.append(file)
                        if not file_already_exist:
                            file_item = File(file_name, file_size)
                            task.input_files.append(file_item)
                            files.append(file_item)

        # 如果节点标签包含 child，则表示这是一个任务之间的依赖关系。
        elif "child" in node.tag.lower():
            child_num = convertTaskRealIdToNum(node.attrib.get("ref"))
            child = getTask(child_num)
            # 在任务之间建立依赖关系。
            for parent in node:
                parent_num = convertTaskRealIdToNum(parent.attrib.get("ref"))
                parent = getTask(parent_num)
                child.pred.append(parent)
                parent.succ.append(child)

    tasks = setTaskEntryAndExit(tasks)
    return tasks, files


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def getAdjAndFeatures(tasks):
    features = []
    for task in tasks:
        if task.num == 0:
            break
        task_features = [
            task.length,
            task.uprank,
            task.rank_exe,
            task.rank_trans,
            # task.status.value,
            task.depth,
            task.depth_len,
            task.input_size,
            task.output_size,
        ]
        features.append(task_features)
    features_matrix = np.array(features)

    num_tasks = len(tasks) - 2
    adj_matrix = np.zeros((num_tasks, num_tasks), dtype=int)
    for i, task in enumerate(tasks):
        if task.num == 0:
            break
        for successor in task.succ:
            j = successor.num - 1
            adj_matrix[i, j] = 1
    adj_coo = sp.coo_matrix(adj_matrix)

    features = normalize_features(features_matrix)
    adj = normalize_adj(adj_coo + sp.eye(adj_coo.shape[0]))

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(features)
    return features, adj
