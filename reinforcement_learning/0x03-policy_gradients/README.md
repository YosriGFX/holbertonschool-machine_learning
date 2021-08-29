## 0x03. Policy Gradients

### 0. Simple Policy function
```
$ cat 0-main.py
#!/usr/bin/env python3
"""
Main file
"""
import numpy as np
from policy_gradient import policy


weight = np.ndarray((4, 2), buffer=np.array([
    [4.17022005e-01, 7.20324493e-01], 
    [1.14374817e-04, 3.02332573e-01], 
    [1.46755891e-01, 9.23385948e-02], 
    [1.86260211e-01, 3.45560727e-01]
    ]))
state = np.ndarray((1, 4), buffer=np.array([
    [-0.04428214,  0.01636746,  0.01196594, -0.03095031]
    ]))

res = policy(state, weight)
print(res)

$
$ ./0-main.py
[[0.50351642 0.49648358]]
$

```
### 1. Compute the Monte-Carlo policy gradient
```
$ cat 1-main.py
#!/usr/bin/env python3
"""
Main file
"""
import gym
import numpy as np
from policy_gradient import policy_gradient

env = gym.make('CartPole-v1')
np.random.seed(1)

weight = np.random.rand(4, 2)
state = env.reset()[None,:]
print(weight)
print(state)

action, grad = policy_gradient(state, weight)
print(action)
print(grad)

env.close()

$ 
$ ./1-main.py
[[4.17022005e-01 7.20324493e-01]
 [1.14374817e-04 3.02332573e-01]
 [1.46755891e-01 9.23385948e-02]
 [1.86260211e-01 3.45560727e-01]]
[[ 0.04228739 -0.04522399  0.01190918 -0.03496226]]
0
[[ 0.02106907 -0.02106907]
 [-0.02253219  0.02253219]
 [ 0.00593357 -0.00593357]
 [-0.01741943  0.01741943]]
$ 

```
### 2. Implement the training
```
$ cat 2-main.py
#!/usr/bin/env python3
"""
Main file
"""
import gym
import matplotlib.pyplot as plt
import numpy as np

from train import train

env = gym.make('CartPole-v1')

scores = train(env, 10000)

plt.plot(np.arange(len(scores)), scores)
plt.show()
env.close()

$ 
$ ./2-main.py

```
### 3. Animate iteration
```
$ cat 3-main.py
#!/usr/bin/env python3
"""
Main file
"""
import gym

from train import train

env = gym.make('CartPole-v1')

scores = train(env, 10000, 0.000045, 0.98, True)

env.close()

$ 
$ ./3-main.py

```

---

> Copyright © 2021 Holberton School. All rights reserved.

![Yosri Ghorbel](https://pbs.twimg.com/media/E3YEO7kXwAU9x6x?format=png&name=4096x4096)
