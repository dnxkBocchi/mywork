import os
import random
import numpy as np
from .dag_generator import workflows_generator


class Workload:
    counter = 0

    def __init__(
        self,
        env,
        workflow_submit_pipe,
        wf_path,
        arrival_rate,
        max_wf_number=float("inf"),
        random_seed=-1,
        debug=False,
        xhn_works=None
    ):
        Workload.counter += 1
        self.id = "wl" + str(Workload.counter)
        self.env = env
        self.workflow_submit_pipe = workflow_submit_pipe
        self.workflow_path = wf_path
        self.arrival_rate = arrival_rate
        # 限制系统中同时存在的工作流数量。
        self.max_wf_number = max_wf_number
        self.debug = debug
        self.rand_seed = random_seed
        self.__submitted_wf_number = 0
        # 用于存储工作流延迟数据
        self.delays = []
        # self.generator = workflows_generator()
        self.generator = xhn_works

        env.process(self.__run())

    # 向工作流管道 self.workflow_submit_pipe 发送DAX的文件路径
    def __run(self):
        while self.__submitted_wf_number < self.max_wf_number:
            random.seed(self.rand_seed + self.__submitted_wf_number)
            np.random.seed(self.rand_seed + self.__submitted_wf_number)

            if self.workflow_path == "generator":
                # 从工作流生成器中随机选择一个工作流
                # wf_path = random.choice(self.generator)
                wf_path = self.generator[self.__submitted_wf_number]
            else:
                self.cached_dax_files = [f for f in os.listdir(self.workflow_path)]
                dax = random.choice(self.cached_dax_files)
                wf_path = self.workflow_path + "/" + dax

            if self.debug:
                print(
                    "[{:.2f} - {:10s}] workflow {} submitted.".format(
                        self.env.now,
                        "Workload",
                        self.__submitted_wf_number
                    )
                )

            self.__submitted_wf_number += 1
            yield self.workflow_submit_pipe.put(wf_path)

        # 当所有工作流提交完成，发送 "end" 信号，表示不再有新的工作流提交。
        yield self.workflow_submit_pipe.put("end")

    @staticmethod
    def reset():
        Workflow.counter = 0
        Workload.counter = 0


class Workflow:
    counter = 0

    def __init__(self, tasks, files=None, path="", submit_time=0):
        Workflow.counter += 1
        self.id = "wf" + str(Workflow.counter)
        self.path = path

        self.fastest_exe_time = 0
        self.deadline_factor = 0
        self.cheapest_exe_cost = 0
        self.budget_factor = 0

        self.deadline = 0
        self.budget = 0
        self.submit_time = submit_time
        self.length = 0
        self.finished_tasks = []

        self.tasks = tasks
        self.files = files
        self.exit_task = None
        self.entry_task = None
        self.features = None
        self.adj_matrix = None
        self.critical_path = None
        self.critical_length = 0

        self.cost = 0
        self.makespan = 0
        self.waiting_time = 0

        for task in tasks:
            task.setWorkflow(self)

            if len(task.pred) == 0:
                self.entry_task = task
            elif len(task.succ) == 0:
                self.exit_task = task

    def getTaskNumber(self):
        return len(self.tasks)

    @staticmethod
    def reset():
        Workflow.counter = 0
