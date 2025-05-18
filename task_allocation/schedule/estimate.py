import math
import random
from collections import defaultdict, deque


def exeTime(task, ncp):
    return task.length / ncp.compute_capacity

def transTime(task, ncp):
    return task.length / ncp.bandwidth

def exeCost(task, ncp):
    # 调用 exeTime 计算任务执行时间，再除以 vm.cycle_time 以确定执行任务需要多少个周期，并用 math.ceil 向上取整。
    return math.ceil(exeTime(task, ncp) / ncp.cycle_time) * ncp.cycle_price


def transferTime(size, bandwidth):
    return size / bandwidth


# 计算给定任务从其父任务传输输入文件所需的最大时间
def rank_maxParentInputTransferTime(task, ncp):
    transfer_size = 0
    # 遍历当前任务的输入文件 task.input_files，检查这些文件是否存在于父任务 p 的输出文件 p.output_files 中。
    # 如果是，将该文件的大小 f.size 累加到 a 中。
    for p in task.pred:
        a = 0
        for f in task.input_files:
            if f in p.output_files:
                a += f.size
        transfer_size = a if a > transfer_size else transfer_size

    # 计算并返回最大传输时间
    return transferTime(transfer_size, ncp.bandwidth)


def maxParentInputTransferTime(task, ncp):
    vms = []
    transfer_time = []
    # 遍历任务的前驱任务（task.pred），识别与当前 VM 不同的 VM，并将其添加到 vms 列表中
    for ptask in task.pred:
        if len(ptask.pred) and ptask.vm().id != ncp.vm.id:
            if ptask.vm not in vms:
                vms.append(ptask.vm)
    for v in vms:
        total_size = 0
        files = task.input_files + []
        for file in files:
            if file in v().disk_items:
                total_size += file.size
                files.remove(file)
        transfer_time.append(v().ncp.transferTime(total_size, ncp, ncp.vm.g))
    return max(transfer_time) if transfer_time else 0


# 越开头越大
def setUpwardRank(task, rank):
    if task.uprank < rank:
        task.uprank = rank

    for ptask in task.pred:
        setUpwardRank(ptask, task.uprank + task.rank_trans + ptask.rank_exe)


# 为工作流 wf 创建截止日期（deadline）。
def createDeadline(wf, fastest_ncp_type, min_df=4, max_df=8, constant_df=0):
    random.seed(50)
    wf.fastest_exe_time = wf.entry_task.uprank

    if constant_df:
        wf.deadline_factor = constant_df
    else:
        wf.deadline_factor = random.randint(min_df, max_df)
    wf.deadline = round(wf.deadline_factor * wf.fastest_exe_time, 2)


def createDeadline2(wf, fastest_ncp_type, args):
    length = wf.critical_length
    action = args.action_num
    task_num = len(wf.tasks)
    fastest_time = length / fastest_ncp_type.compute_capacity
    wf.deadline = round(fastest_time, 2)


# 为工作流 wf 计算预算（budget）。
def createBudget(
    wf, cheapest_ncp_type, min_bf=1, max_bf=20, factor_int=True, constant_bf=0
):
    total_time = 0
    for task in wf.tasks:
        total_time += exeTime(task, cheapest_ncp_type)

    cycle_num = math.ceil(total_time / cheapest_ncp_type.cycle_time)
    wf.cheapest_exe_cost = cycle_num * cheapest_ncp_type.cycle_price

    if constant_bf:
        wf.budget_factor = constant_bf
    else:
        wf.budget_factor = (
            random.randint(min_bf, max_bf)
            if factor_int
            else random.uniform(min_bf, max_bf)
        )
    wf.budget = round(wf.budget_factor * wf.cheapest_exe_cost, 2)


# 为任务列表中的每个任务计算其在不同虚拟机（VM）上的执行时间和执行成本，
def RunTimeCost(task_list, NCPs):
    for task in task_list:
        # 计算所有虚拟机的时间成本
        task.vm_time_cost = {}
        for ncp in NCPs:
            tran_exe_time = (
                exeTime(task, ncp)
                + transTime(task, ncp)
                + ncp.vm.waiting_time  # 必须是vm此时的等待时间， 不能是task的等待时间，因为task还没开始计算呢
            )
            cost = exeCost(task, ncp)
            # dict.update() 方法会更新字典，如果字典中已经存在相同的键，则覆盖旧值，如果不存在该键，则添加新键值对。
            task.vm_time_cost.update({ncp: [tran_exe_time, cost]})

        # 确保任务优先选择执行时间最短的虚拟机，并重新生成一个有序的字典，代码没错
        task.vm_time_cost = dict(
            sorted(task.vm_time_cost.items(), key=lambda item: item[1][0])
        )
        # 表示在任务上最快运行的虚拟机的执行时间
        task.fast_run = list(task.vm_time_cost.values())[0][0]
