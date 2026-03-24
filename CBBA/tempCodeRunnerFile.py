def print_metrics(env):
    uavs = env.uavs
    targets = env.targets
    for uav in uavs:
        print(f"{uav.id}: tasks={uav.tasks}")
    for uav in uavs:
        distance = uav._init_voyage - uav.voyage
        print(f"{uav.id}: voyage={distance:.2f}")
    for target in targets:
        print(f"{target.id}: finish_time={target.total_time:.2f}") 