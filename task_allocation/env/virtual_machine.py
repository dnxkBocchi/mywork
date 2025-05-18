import weakref
import simpy

from .task import TaskStatus
from schedule.estimate import exeTime, maxParentInputTransferTime, transTime


class VirtualMachine:
    counter = 0

    def __init__(self, env, ncp, NCP_graph, debug=False):
        VirtualMachine.counter += 1
        self.id = "vm" + str(ncp.node_id)
        self.num = VirtualMachine.counter + 0
        self.debug = debug
        self.env = env
        self.ncp = ncp
        self.g = NCP_graph

        self.running = False
        self.workload_finished = False

        self.start_time = 0
        self.release_time = 0
        self.tasks = []
        self.done_tasks = []
        # 磁盘，记录输入文件和输出文件
        self.disk_items = []

        self.unfinished_tasks_number = 0
        self.waiting_time = 0
        self.finished_tasks_number = 0
        self.task_length = 0
        self.task_number = 0

        # The Store operates in a FIFO (first-in, first-out) order
        self.task_queue = simpy.Store(env)
        self.task_finished_announce_pipe = None
        self.vm_release_announce_pipe = None

    def start(self):
        self.env.process(self.__start())

    # 检查 VM 是否处于空闲状态
    def __checkIdle(self):
        while self.running:
            yield self.env.timeout(self.ncp.cycle_time)
            if self.isIdle() and self.workload_finished:
                self.vm_release_announce_pipe.put(self)
                self.release_time = self.env.now
                self.running = False

    # 估算给定任务的完成时间
    def estimateFinishTime(self, task):
        # 确定最大传输时间
        task.trans_time = transTime(task, self.ncp)
        task.exe_time = exeTime(task, self.ncp)
        task.estimate_finish_time = (
            self.env.now + task.trans_time + task.exe_time + self.waiting_time
        )

    # 提交任务到 VM 的任务队列
    def submitTask(self, task):
        self.unfinished_tasks_number += 1
        self.tasks.append(task)
        self.estimateFinishTime(task)
        self.waiting_time += task.trans_time + task.exe_time
        self.task_length += task.length
        self.task_number += 1
        task.workflow.waiting_time += self.waiting_time
        # 更新磁盘项，将任务输出的文件添加到磁盘项中。
        self.disk_items += task.output_files
        self.disk_items += task.input_files

        yield self.task_queue.put(task)
        task.vm_queue_time = self.env.now
        task.status = TaskStatus.wait
        # 保存 VM 引用， 创建对对象的弱引用
        task.vm = weakref.ref(self)

        if self.debug:
            print(
                "[{:.2f} - {:10s}] {} task is submitted to vm queue, queue waiting task size {}.".format(
                    self.env.now,
                    self.id,
                    task.id,
                    self.unfinished_tasks_number,
                )
            )

    # 管理任务的状态和资源的使用
    def __exeProcess(self, task):
        task.status = TaskStatus.run
        if self.debug:
            print(
                "[{:.2f} - {:10s}] {} task is start executing.".format(
                    self.env.now, self.id, task.id
                )
            )
        yield self.env.timeout(task.exe_time)
        task.finish_time = self.env.now
        self.done_tasks.append(task)
        self.finished_tasks_number += 1
        self.unfinished_tasks_number -= 1
        self.waiting_time -= (task.trans_time + task.exe_time)
        self.task_finished_announce_pipe.put(task)
        self.ncp.cpu -= 0.07

        if self.debug:
            print(
                "[{:.2f} - {:10s}] {} task is finished, rank_trans:{:.2f}, rank_exe:{:.2f}, trans:{:.2f}, exe:{:.2f},  waiting time:{:.2f}, use time: {:.2f}.".format(
                    self.env.now,
                    self.id,
                    task.id,
                    task.rank_trans,
                    task.rank_exe,
                    task.trans_time,
                    task.exe_time,
                    task.start_time - task.vm_queue_time,
                    task.finish_time - task.start_time,
                )
            )

    # 从任务队列中获取任务并执行
    def __cpu(self):
        while self.running:
            task = yield self.task_queue.get()
            # I/O
            task.start_time = self.env.now
            if task.trans_time:
                yield self.env.timeout(task.trans_time)
            # CPU
            yield self.env.process(self.__exeProcess(task))

    # 生成器，用于启动虚拟机（VM）的过程
    def __start(self):
        yield self.env.timeout(self.ncp.startup_delay)
        self.start_time = self.env.now
        self.running = True
        self.env.process(self.__checkIdle())
        self.env.process(self.__cpu())

    def runningTime(self):
        if self.running:
            return self.env.now - self.start_time
        return 0

    def isVMncp(self):
        return False

    def isIdle(self):
        return self.running and self.unfinished_tasks_number == 0

    @staticmethod
    def reset():
        VirtualMachine.counter = 0

    def __repr__(self):
        return "{}".format(self.id)
