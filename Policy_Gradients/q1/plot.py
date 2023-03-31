import matplotlib.pyplot as plt
import numpy as np

maxepisode = 500
maxiter = 200
part = 1

def execute():
    global part
    global maxepisode
    global maxepisode

    print("execute")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if part == 1:
        flist = ["part_1_ep_500.npy"]
        for fn in  flist:
            rewards = np.load(fn)
            iter = np.arange(1, len(rewards) + 1)
            plt.plot(iter, rewards)
    if part == 2:
        flist = ["part_1_ep_500.npy", "part_2_ep_500.npy"]
        for fn in  flist:
            rewards = np.load(fn)
            iter = np.arange(1, len(rewards) + 1)
            plt.plot(iter, rewards)
        plt.legend(["part1", "part2"])
    if part == 3:
        flist = ["part_1_ep_500.npy", "part_2_ep_500.npy", "part_3_ep_500.npy"]
        for fn in  flist:
            rewards = np.load(fn)
            iter = np.arange(1, len(rewards) + 1)
            plt.plot(iter, rewards)
        plt.legend(["part1", "part2", "part3"])
    if part == 4:
        flist = ["part_4_ep_100.npy", "part_4_ep_300.npy", "part_4_ep_1000.npy"]
        # flist = ["part_4_ep_100.npy"]
        for fn in flist:
            rewards = np.load(fn)
            iter = np.arange(1, len(rewards) + 1)
            plt.plot(iter, rewards)
        plt.legend(["100 Ep", "300 Ep", "1000 Ep"])

    plt.ylabel('Average Reward')
    plt.xlabel('Iteration')
    plt.title('Q1 part ' + str(part))
    plt.savefig("./part_" + str(part) +".jpg", dpi = 200)

for i in range(1, 5):
    part = i
    print("Start: part " + str(part))
    execute()