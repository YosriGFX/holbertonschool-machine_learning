# 0x07. Bayesian Probability


![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/8/8358e1144bbb1fcc51b4.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210609%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210609T141726Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=1b46e1ea940a0bccde01093e84a236ae81717336a564fda779e3daec02c061fa)

### 0. Likelihood
```
alexa@ubuntu-xenial:0x07-bayesian_prob$ cat 0-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    likelihood = __import__('0-likelihood').likelihood

    P = np.linspace(0, 1, 11) # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(likelihood(26, 130, P))
alexa@ubuntu-xenial:0x07-bayesian_prob$ ./0-main.py 
[0.00000000e+00 2.71330957e-04 8.71800070e-02 3.07345706e-03
 5.93701546e-07 1.14387595e-12 1.09257177e-20 6.10151799e-32
 9.54415702e-49 1.00596671e-78 0.00000000e+00]
alexa@ubuntu-xenial:0x07-bayesian_prob$
```

### 1. Intersection
```
alexa@ubuntu-xenial:0x07-bayesian_prob$ cat 1-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    intersection = __import__('1-intersection').intersection

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11 # this prior assumes that everything is equally as likely
    print(intersection(26, 130, P, Pr))
alexa@ubuntu-xenial:0x07-bayesian_prob$ ./1-main.py 
[0.00000000e+00 2.46664506e-05 7.92545518e-03 2.79405187e-04
 5.39728678e-08 1.03988723e-13 9.93247059e-22 5.54683454e-33
 8.67650639e-50 9.14515194e-80 0.00000000e+00]
alexa@ubuntu-xenial:0x07-bayesian_prob$
```

### 2. Marginal Probability
```
alexa@ubuntu-xenial:0x07-bayesian_prob$ cat 2-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    marginal = __import__('2-marginal').marginal

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(marginal(26, 130, P, Pr))
alexa@ubuntu-xenial:0x07-bayesian_prob$ ./2-main.py 
0.008229580791426582
alexa@ubuntu-xenial:0x07-bayesian_prob$
```

### 3. Posterior
```
alexa@ubuntu-xenial:0x07-bayesian_prob$ cat 3-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    posterior = __import__('3-posterior').posterior

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(posterior(26, 130, P, Pr))
alexa@ubuntu-xenial:0x07-bayesian_prob$ ./3-main.py 
[0.00000000e+00 2.99729127e-03 9.63044824e-01 3.39513268e-02
 6.55839819e-06 1.26359684e-11 1.20692303e-19 6.74011797e-31
 1.05430721e-47 1.11125368e-77 0.00000000e+00]
alexa@ubuntu-xenial:0x07-bayesian_prob$
```

### 4. Continuous Posterior
```
alexa@ubuntu-xenial:0x07-bayesian_prob$ cat 100-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    posterior = __import__('100-continuous').posterior

    print(posterior(26, 130, 0.17, 0.23))
alexa@ubuntu-xenial:0x07-bayesian_prob$ ./100-main.py 
0.6098093274896035
alexa@ubuntu-xenial:0x07-bayesian_prob$
```

> Copyright © 2021 Holberton School. All rights reserved.

![Yosri Ghorbel](https://pbs.twimg.com/media/E3YEO7kXwAU9x6x?format=png&name=4096x4096)

