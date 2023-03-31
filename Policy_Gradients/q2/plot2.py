import matplotlib.pyplot as plt
import numpy as np

maxepisode = 500
maxiter = 200

print("execute")

fig = plt.figure()
ax = fig.add_subplot(111)

flist = ["q2_ep_30.npy", "q2_ep_60.npy", "q2_ep_100.npy"]

for fn in  flist:
    rewards = np.load(fn)
    iter = np.arange(1, len(rewards) + 1)
    plt.plot(iter, rewards)

plt.legend(["30 Ep", "60 Ep", "100 Ep"])
plt.ylabel('Average Reward')
plt.xlabel('Iteration')
plt.title('Q2')
plt.savefig("./q2"".jpg", dpi = 200)