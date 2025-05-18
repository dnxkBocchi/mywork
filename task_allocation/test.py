from env.workload import Workload, Workflow
from schedule.taskNcpEnv import TaskAndNcpEnvironment
from schedule.schedule import Scheduler
from model.hyperparameter import get_args
from model.dqn import DQN
from model.sparse_gat import SpGAT
import datetime
import time
import pickle

from server_client import *
from threading import Thread
from smart_node import real_node

args = get_args()
args.debug = False

def test(agent, test_wf_path, model, model_path, xhn_works, xhn_nodes):
    mean_makespan = []
    load_balance = []
    time_rate = []

    model.train(False)
    agent.load_model(model_path)
    agent.net.train(False)
    scheduler = Scheduler(agent, model, args)
    print("start at:", str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    start = time.time()

    num = 0
    for episode in range(1, args.test_number+1):
        Workflow.reset()
        Workload.reset()

        print("episode:",episode,"="*70)
        t, lb, tr, decision_making = TaskAndNcpEnvironment(test_wf_path, scheduler, episode*10, args, method="dqn", 
                xhn_works=xhn_works, xhn_nodes=xhn_nodes, destroy_ncp=None, destroy_time=3000, output=True)
        print(decision_making)
        mean_makespan.append(t)
        load_balance.append(lb)
        time_rate.append(tr)

        num += tr

    s = str(datetime.timedelta(seconds=time.time()-start))
    print("total train time:", s, "\n", "total time finish:", num)

    return decision_making


def deal_tasks_hr(xhn_works, xhn_nodes, client):
    generator = 'generator'
    model_path_5node = "D:/code/python_project/zgq/load_balancing/save_model/dqn_model_generator_5node.pth"
    
    model = SpGAT(args.nfeat, args.hidden, args.out_feature, args.dropout, args.alpha, args.nb_heads).to(args.device)
    
    agent1 = DQN(args, model)
    
    decision_making = test(agent1, generator, model, model_path_5node, xhn_works, xhn_nodes)
    
    msg_to_send=message()
    msg_to_send.content=decision_making
    data = pickle.dumps(msg_to_send)
    send_to_server(client, data)


client = build_recv_server("0.0.0.0", 5008)

my_threads=[]
while True:
    xhn_works = []
    while True:
        client_socket, client_address = client.accept()
        data = recv_from_server(client_socket)
        msg = pickle.loads(data)
        xhn_work = msg.content
        xhn_works.append(xhn_work)
        if len(xhn_works) == args.wf_number:
            print("end")
            break
        

    query_msg = message()
    query_msg.type = "get_nodes_info"
    query_data = pickle.dumps(query_msg)
    client1 = build_send_client("192.168.27.130", 5002)
    send_to_server(client1, query_data)
    ans = recv_from_server(client1)
    ans_tmp = pickle.loads(ans)
    xhn_nodes = ans_tmp.content

    thread = Thread(target = deal_tasks_hr, args=(xhn_works, xhn_nodes[0:args.action_num], client_socket))
    my_threads.append(thread)
    thread.start()