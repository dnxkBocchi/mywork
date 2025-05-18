import random
import os
import torch
import numpy as np
import simpy
import matplotlib.pyplot as plt

from env.workload import Workload, Workflow
from env.task import (
    TaskStatus,
    parseDAX,
    parse_generate_dag,
    parse_xhn_tasks,
    getAdjAndFeatures,
    find_critical_path,
)
from env.virtual_machine import VirtualMachine
from env.ncp_network import create_ncp_graph, create_NCP_network, create_xhn_ncps, Node
from schedule import estimate


def setRandSeed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def NcpVmEnvironment(sim, debug, action_num, nodes):
    # NCP_graph, NCPs = create_ncp_graph(node_nums=action_num)
    NCP_graph, NCPs = create_xhn_ncps(nodes)
    vms = []
    for ncp in NCPs:
        vm = VirtualMachine(sim, ncp, NCP_graph, debug=debug)
        vm.start()
        ncp.vm = vm
        vms.append(vm)
    return NCPs, vms


def TaskAndNcpEnvironment(wf_path, Scheduler, seed, args, method=None, 
                        xhn_works=None, xhn_nodes=None, 
                        destroy_ncp=None, destroy_time=0, 
                        increase_task=None, increase_time=0,
                        output=False):
    global remained_tasks, workload_finished, running, capacity_fault, time_step, des_ncp, des_time, max_cpu_rate
    running = True
    remained_tasks = 0
    workload_finished = False
    capacity_fault = 0
    time_step = 0
    max_cpu_rate = 0.0

    des_ncp = destroy_ncp
    des_time = destroy_time

    debug = args.debug
    wf_arrival_rate = args.arrival_rate
    finished_wfs = []
    tasks_ready_queue = []
    decision_making = []

    sim = simpy.Environment()
    workflow_submit_pipe = simpy.Store(sim)
    task_finished_announce_pipe = simpy.Store(sim)
    vm_release_announce_pipe = simpy.Store(sim)
    ready_queue_key = simpy.Resource(sim, 1)
    ready_task_counter = simpy.Container(sim, init=0)

    setRandSeed(seed * 5)
    NCPs, vms = NcpVmEnvironment(sim, debug, args.action_num, xhn_nodes)
    fastest_ncp_type = max(NCPs, key=lambda v: v.compute_capacity)
    cheapest_ncp_type = min(NCPs, key=lambda v: v.cycle_price)

    Workload(
        sim, workflow_submit_pipe, wf_path, wf_arrival_rate, args.wf_number, seed, debug, xhn_works
    )

    def __poolingProcess():
        global running, workload_finished, remained_tasks
        while running and not workload_finished:
            dag = yield workflow_submit_pipe.get()
            if dag == "end":
                workload_finished = True
                return

            if wf_path == "generator":
                # tasks, files = parse_generate_dag(dag)
                tasks, files = parse_xhn_tasks(dag)
                # wf = Workflow(tasks, path=dag[3], submit_time=sim.now)
                wf = Workflow(tasks, submit_time=sim.now)
                
            else:
                tasks, files = parseDAX(dag)
                wf = Workflow(tasks, path=dag, submit_time=sim.now)

            wf.critical_path, wf.critical_length = find_critical_path(tasks)
            # 初始在最快的ncp上计算一下传输时间和执行时间
            for task in wf.tasks:
                task.status = TaskStatus.pool
                task.rank_trans = estimate.transTime(task, fastest_ncp_type)
                task.rank_exe = estimate.exeTime(task, fastest_ncp_type)
                wf.length += task.rank_exe
            # 计算每个任务的上行优先级
            estimate.setUpwardRank(wf.exit_task, 0)
            estimate.createDeadline2(wf, fastest_ncp_type, args)
            for task in wf.tasks:
                task.deadline = task.rank_exe / wf.length * wf.deadline

            remained_tasks += len(wf.tasks) - 2
            wf.entry_task.status = TaskStatus.done

            __addToReadyQueue(wf.entry_task.succ)
            yield ready_task_counter.put(1)

            if debug:
                print(
                    "[{:.2f} - {:10s}] {} (id: {}, task sums: {}, deadline: {:.2f}, budget: {:.2f}, df: {:.2f}, bf: {:.2f}) is saved in the pool.".format(
                        sim.now,
                        "Pool",
                        wf.path,
                        wf.id,
                        len(wf.tasks),
                        wf.deadline,
                        wf.budget,
                        wf.deadline_factor,
                        wf.budget_factor,
                    )
                )

    # 将一组任务（task_list）添加到一个准备执行的任务队列（tasks_ready_queue）中，并更新这些任务的状态和准备时间。
    def __addToReadyQueue(task_list):
        for t in task_list:
            t.status = TaskStatus.ready
            t.ready_time = sim.now
        request_key = ready_queue_key.request()
        tasks_ready_queue.extend(task_list)
        ready_queue_key.release(request_key)

        if debug:
            print(
                "[{:.2f} - {:10s}] {} tasks are added to ready queue. queue size: {}.".format(
                    sim.now, "ReadyQueue", len(task_list), len(tasks_ready_queue)
                )
            )

    # 监听任务完成的消息，并根据任务的状态更新工作流的状态,以及训练DQN。
    def __queueingProcess():
        global running, workload_finished, remained_tasks, time_step
        while running:
            finished_task = yield task_finished_announce_pipe.get()
            finished_task.status = TaskStatus.done
            wf = finished_task.workflow
            wf.finished_tasks.append(finished_task)

            ready_tasks = []
            # 遍历完成任务的后继任务（succ）来检查其是否准备好进行调度
            for child in finished_task.succ:
                if child.isReadyToSch():
                    if child != wf.exit_task:
                        ready_tasks.append(child)
                    else:
                        wf.exit_task.status = TaskStatus.done
                        wf.makespan = sim.now - wf.submit_time
                        finished_wfs.append(wf)

            yield sim.timeout(0.2)
            if ready_tasks:
                __addToReadyQueue(ready_tasks)
                yield ready_task_counter.put(1)

    # 负责在虚拟机完成任务后，释放该虚拟机并更新相关的状态
    def __releasingProcess():
        global running
        while running:
            vm = yield vm_release_announce_pipe.get()
            # if debug:
            print(
                "[{:.2f} - {:10s}] {} virtual machine is released. vm tasks num = {}. ".format(
                    sim.now,
                    "Releaser",
                    vm.id,
                    vm.finished_tasks_number,
                )
            )

    # 动态销毁一个虚拟机，并将该虚拟机上的任务重新添加到任务等待队列
    def destroy_ncp_retask(ncp):
        # 随机选择一个虚拟机坠毁，并将该虚拟机上的任务重新添加到任务等待队列
        # ncp_fail = random.choice(ncp_list)
        ncp_failed = NCPs[1]
        ncp_failed.fail = True
        ncp_failed.running = False
        ncp_failed.compute_capacity = 1
        # 未完善，任务完成就不需要重新添加
        vm_failed = ncp_failed.vm
        print("vm_failed.tasks finished: ", vm_failed.finished_tasks_number)
        if debug:
            print(
                "[{:.2f} - {:10s}] {} ncp virtual machine is failed.".format(
                    sim.now, "Failure", ncp_failed.node_id
                )
            )
        if vm_failed.unfinished_tasks_number:
            __addToReadyQueue(vm_failed.tasks)
            ready_task_counter.put(1)

    # 动态增加任务
    def increase_task_dynamic():
        global workload_finished
        workload_finished = False
        Workload(
            sim,
            workflow_submit_pipe,
            wf_path,
            wf_arrival_rate,
            2,
            seed,
            debug,
        )

    # 选择VM
    def chooseNCP(ncp_list, chose_task):
        global remained_tasks, workload_finished, time_step
        time_step += 1

        if method == "random":
            return random.choice(ncp_list)
        elif method == "rotation":
            return ncp_list[time_step % len(ncp_list)]
        elif method == "dqn":
            action = Scheduler.runDQN(
                chose_task,
                ncp_list,
                vms,
                remained_tasks == 0 and workload_finished,
            )
            return ncp_list[action]
        elif method == "ppo":
            action = Scheduler.runPPO(
                chose_task,
                ncp_list,
                vms,
                remained_tasks == 0 and workload_finished,
            )
            return ncp_list[action]

    # 找到所选的nvm，并且更新其他vm的管道
    def updateVMs(chose_ncp):
        global workload_finished, remained_tasks
        for i in range(len(vms)):
            if vms[i].ncp == chose_ncp:
                nvm = vms[i]
                break
        # 设置虚拟机的管道，用于在任务完成后通知任务调度器，以及在虚拟机资源释放后进行通信。
        for vmi in vms:
            vmi.task_finished_announce_pipe = task_finished_announce_pipe
            vmi.vm_release_announce_pipe = vm_release_announce_pipe
            if workload_finished and remained_tasks == 0:
                vmi.workload_finished = True
        return nvm

    def __schedulingProcess():
        global running, remained_tasks, capacity_fault, time_step, des_ncp, des_time, max_cpu_rate
        while running:
            yield ready_task_counter.get(1)
            while len(tasks_ready_queue):
                estimate.RunTimeCost(tasks_ready_queue, NCPs)
                # 对任务列表排序,按照截止时间-最快时间，剩余时间如果是负，那更紧急了
                tasks_ready_queue.sort(key=lambda t: t.deadline - t.fast_run)
                chose_task = tasks_ready_queue.pop(0)
                remained_tasks -= 1

                chose_ncp = chooseNCP(NCPs, chose_task)
                chose_ncp.cpu += 0.07
                max_cpu_rate = max(max_cpu_rate, chose_ncp.cpu)

                if output:
                    decision_making.append(
                        {
                            "task": chose_task.num,
                            "ncp": chose_ncp.node_id,
                        }
                    )
                
                if remained_tasks + 500 == 0:
                    cpu_list = []
                    for ncp in NCPs:
                        cpu_list.append(ncp.cpu)
                    # 绘制折线图
                    plt.plot(cpu_list)

                    # 添加标题和标签
                    plt.title('CPU Utilization of NCPs')
                    plt.xlabel('NCP Index')
                    plt.ylabel('CPU Utilization')

                    # 显示图形
                    plt.show()
                    

                if debug:
                    print(
                        "[{:.2f} - {:10s}] {} task chose for scheduling. L:{:.2f} , trans time:{:.2f}, exe time:{:.2f}, predict time :{:.2f}, deadline :{:.2f} \n \
                            to nvm: {}, cpu: {:.2f}, no {} task.".format(
                            sim.now,
                            "Scheduler",
                            chose_task.id,
                            chose_task.length,
                            chose_task.rank_trans,
                            chose_task.rank_exe,
                            chose_task.vm_time_cost[chose_ncp][0],
                            chose_task.deadline,
                            chose_ncp.node_id,
                            chose_ncp.compute_capacity,
                            time_step,
                        )
                    )

                chose_task.workflow.cost += chose_task.vm_time_cost[chose_ncp][1]
                chose_task.vm_time_cost = {}
                nvm = updateVMs(chose_ncp)
                yield sim.process(nvm.submitTask(chose_task))

    def lastFunction():
        global max_cpu_rate
        total_time = 0.0
        total_cost = 0.0
        load_balance = []
        budget_meet = 0.0
        deadline_meet = 0.0
        both_meet = 0.0

        load_balance = np.array([vm.task_length for vm in vms])
        load_balance = (load_balance - load_balance.min()) / (
            load_balance.max() - load_balance.min()
        )

        for wf in finished_wfs:
            total_time += wf.makespan
            total_cost += wf.cost

            if wf.cost <= wf.budget:
                budget_meet += 1
            if wf.makespan <= wf.deadline:
                deadline_meet += 1
            print(
                "[Deadline] wf.path: {}, wf.deadline = {:.2f}, wf.makespan = {:.2f}, wf.waiting_time = {:.2f}, distance = {:.2f}, num = {}".format(
                    wf.path,
                    wf.deadline,
                    wf.makespan,
                    wf.waiting_time,
                    wf.deadline - wf.makespan,
                    len(wf.tasks),
                )
            )

            if wf.cost <= wf.budget and wf.makespan <= wf.deadline:
                both_meet += 1

        print(
            "cost fail total : {}, makespan fail total : {}, capacity_fault: {}, max_cpu_rate: {}".format(
                len(finished_wfs) - budget_meet,
                len(finished_wfs) - deadline_meet,
                capacity_fault,
                max_cpu_rate
            )
        )
        total_time /= len(finished_wfs)
        total_cost /= len(finished_wfs)
        var_load_balance = np.var(load_balance)
        budget_meet /= len(finished_wfs)
        deadline_meet /= len(finished_wfs)
        both_meet /= len(finished_wfs)
        return (
            total_time,
            var_load_balance,
            deadline_meet,
            decision_making,
        )

    sim.process(__poolingProcess())
    sim.process(__schedulingProcess())
    sim.process(__queueingProcess())
    sim.process(__releasingProcess())

    # 启动仿真，执行所有已经注册到 SimPy 环境中的进程和事件。
    sim.run()
    return lastFunction()
