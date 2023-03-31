There are q1 and q2 directories inside the package.

Both are the implementation of the vanilla reinforcement learning using a neural network policy module.

q1 is an implementation on the cartpole and q2 is on the link arm.

In order to execute all parts in q1, run the following in the q1 directory.

```
python q1.py
```

In order to execute q2, run the following in the q2 directory.

```
python q2.py
```

To render a one episode with the trained model, change the parameter, set render_flag == True and run q2.py as above. Also, the parameter, pi_path and optimizer_path, must be set to the saved trained model path.

Installation:
```
pip install gym==0.21.0
git clone https://github.com/benelot/pybullet-gym.git
pip install -e pybullet-gym/
pip install pybullet == 3.2.1
git clone https://github.com/ucsdarclab/pybullet-gym-env.git
pip install -e pybullet-gym-env/
pip install -e modified-gym-env/
```

## Usage

Once the package is installed, you can create an environment using the following command:

```
env = gym.make("CartPole-v1", rand_init=False)
```

If `rand_init=True`, then the arm will initialize to a random location after each reset.