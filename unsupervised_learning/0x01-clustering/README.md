# 0x01. Clustering

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/8/95b077928a7de094f94f.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210609%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210609T221402Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=9608c0e80bf1ca417aa98dca1d1328f6f04b4b767f27a72d2a54c262af813b97)


### 0. Initialize K-means
```
yosri@ubuntu-xenial:0x01-clustering$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
initialize = __import__('0-initialize').initialize

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.show()
    print(initialize(X, 5))
yosri@ubuntu-xenial:0x01-clustering$ ./0-main.py

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/8/a5311b6ad0410a678aab.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210609%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210609T221402Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=a7952601281feecbd5a30925a83ca3525f83ae7b4cf8b5b5ec140ab10da6bf2f)

```
[[14.54730144 13.46780434]
 [20.57098466 33.55245039]
 [ 9.55556506 51.51143281]
 [48.72458008 20.03154959]
 [25.43826106 60.35542243]]
yosri@ubuntu-xenial:0x01-clustering$

```

### 1. K-means
```
yosri@ubuntu-xenial:0x01-clustering$ cat 1-main.py
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
kmeans = __import__('1-kmeans').kmeans

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    C, clss = kmeans(X, 5)
    print(C)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.scatter(C[:, 0], C[:, 1], s=50, marker='*', c=list(range(5)))
    plt.show()
yosri@ubuntu-xenial:0x01-clustering$ ./0-main.py
[[ 9.92511389 25.73098987]
 [30.06722465 40.41123947]
 [39.62770705 19.89843487]
 [59.22766628 29.19796006]
 [20.0835633  69.81592298]]

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/8/74c9c16e29333b39f7db.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210609%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210609T221402Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=f495b89eebde8cfe4154cc860c10846dc0ae7e9ecd5b6f3c6a1b0fc572535b91)

### 2. Variance
```
yosri@ubuntu-xenial:0x01-clustering$ cat 2-main.py
#!/usr/bin/env python3

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    for k in range(1, 11):
        C, _ = kmeans(X, k)
        print('Variance with {} clusters: {}'.format(k, variance(X, C).round(5)))
yosri@ubuntu-xenial:0x01-clustering$ ./2-main.py
Variance with 1 clusters: 157927.7052
Variance with 2 clusters: 82095.68297
Variance with 3 clusters: 34784.23723
Variance with 4 clusters: 23158.40095
Variance with 5 clusters: 7868.521233
Variance with 6 clusters: 7406.930773
Variance with 7 clusters: 6930.663613
Variance with 8 clusters: 6162.158842
Variance with 9 clusters: 5843.92455
Variance with 10 clusters: 5727.41124
yosri@ubuntu-xenial:0x01-clustering$

```

### 3. Optimize k
```
yosri@ubuntu-xenial:0x01-clustering$ cat 3-main.py
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
optimum_k = __import__('3-optimum').optimum_k

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    results, d_vars = optimum_k(X, kmax=10)
    print(results)
    print(np.round(d_vars, 5))
    plt.scatter(list(range(1, 11)), d_vars)
    plt.xlabel('Clusters')
    plt.ylabel('Delta Variance')
    plt.title('Optimizing K-means')
    plt.show()
yosri@ubuntu-xenial:0x01-clustering$ ./3-main.py
[(array([[31.78625503, 37.01090945]]), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0])), (array([[34.76990289, 28.71421162],
       [20.14417812, 69.38429903]]), array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
       1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0, 0, 0, 0])), (array([[49.55185774, 24.76080087],
       [20.0835633 , 69.81592298],
       [19.8719982 , 32.85851127]]), array([2, 2, 1, 2, 0, 2, 2, 0, 1, 0, 0, 2, 1, 0, 1, 2, 2, 0, 0, 1, 2, 2,
       1, 0, 2, 2, 0, 0, 2, 0, 0, 2, 1, 2, 0, 2, 0, 2, 0, 1, 1, 0, 2, 2,
       0, 2, 2, 2, 1, 0, 0, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0,
       2, 0, 1, 0, 1, 0, 2, 2, 2, 2, 1, 2, 2, 1, 0, 2, 0, 1, 2, 0, 2, 2,
       0, 1, 2, 2, 0, 2, 2, 1, 2, 1, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0,
       2, 2, 0, 1, 0, 2, 2, 1, 2, 2, 0, 0, 2, 1, 0, 1, 1, 0, 1, 2, 2, 2,
       0, 1, 0, 2, 2, 0, 2, 0, 2, 0, 1, 0, 1, 1, 0, 2, 2, 1, 0, 2, 0, 0,
       1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 1, 1, 1, 1, 2, 0, 0, 1,
       2, 0, 0, 2, 0, 0, 2, 1, 2, 1, 2, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0,
       2, 2, 0, 0, 0, 0, 0, 1, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2,
       1, 0, 0, 2, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0,
       2, 1, 0, 2, 2, 0, 2, 0])), (array([[39.57566544, 20.48452248],
       [20.0835633 , 69.81592298],
       [19.62313956, 33.02895961],
       [59.22766628, 29.19796006]]), array([2, 2, 1, 2, 3, 2, 2, 0, 1, 0, 0, 2, 1, 0, 1, 2, 0, 3, 0, 1, 2, 2,
       1, 3, 2, 2, 0, 0, 2, 3, 3, 2, 1, 2, 0, 2, 3, 2, 3, 1, 1, 0, 2, 2,
       0, 2, 2, 2, 1, 3, 3, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 3,
       2, 0, 1, 3, 1, 0, 2, 2, 2, 2, 1, 2, 2, 1, 3, 2, 3, 1, 2, 0, 2, 2,
       3, 1, 2, 2, 3, 2, 2, 1, 2, 1, 0, 3, 2, 2, 3, 0, 0, 2, 2, 0, 3, 0,
       2, 2, 0, 1, 0, 2, 2, 1, 2, 2, 0, 0, 2, 1, 0, 1, 1, 3, 1, 2, 2, 2,
       3, 1, 3, 2, 2, 0, 2, 3, 2, 3, 1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 0,
       1, 1, 1, 3, 3, 0, 2, 2, 2, 0, 0, 3, 0, 2, 1, 1, 1, 1, 2, 0, 3, 1,
       2, 3, 3, 2, 0, 0, 2, 1, 2, 1, 0, 0, 1, 1, 3, 0, 3, 2, 0, 0, 3, 3,
       2, 2, 3, 0, 0, 0, 3, 1, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2,
       1, 0, 0, 2, 1, 0, 3, 0, 3, 1, 1, 0, 0, 3, 0, 3, 0, 2, 2, 3, 3, 3,
       2, 1, 0, 2, 2, 0, 2, 0])), (array([[30.06722465, 40.41123947],
       [59.22766628, 29.19796006],
       [ 9.92511389, 25.73098987],
       [20.0835633 , 69.81592298],
       [39.62770705, 19.89843487]]), array([0, 0, 3, 0, 1, 2, 0, 4, 3, 4, 4, 0, 3, 4, 3, 2, 4, 1, 4, 3, 2, 0,
       3, 1, 0, 2, 4, 4, 2, 1, 1, 0, 3, 2, 4, 0, 1, 0, 1, 3, 3, 4, 0, 2,
       4, 0, 0, 0, 3, 1, 1, 3, 0, 3, 0, 2, 0, 2, 0, 0, 2, 2, 1, 2, 1, 1,
       0, 4, 3, 1, 3, 4, 0, 0, 2, 0, 3, 2, 0, 3, 1, 0, 1, 3, 2, 4, 2, 0,
       1, 3, 0, 2, 1, 0, 0, 3, 2, 3, 4, 1, 0, 2, 1, 4, 4, 0, 2, 4, 1, 4,
       2, 2, 4, 3, 4, 2, 2, 3, 2, 2, 4, 4, 0, 3, 0, 3, 3, 1, 3, 2, 0, 2,
       1, 3, 1, 2, 0, 4, 0, 1, 0, 1, 3, 1, 3, 3, 1, 2, 2, 3, 1, 2, 1, 4,
       3, 3, 3, 1, 1, 4, 2, 2, 0, 4, 4, 1, 4, 2, 3, 3, 3, 3, 0, 4, 1, 3,
       2, 1, 1, 2, 4, 4, 2, 3, 2, 3, 4, 4, 3, 3, 1, 4, 1, 0, 4, 4, 1, 1,
       2, 0, 1, 4, 4, 0, 1, 3, 1, 2, 0, 1, 0, 2, 0, 2, 2, 0, 0, 2, 3, 0,
       3, 4, 4, 2, 3, 4, 1, 4, 1, 3, 3, 4, 4, 1, 4, 1, 4, 0, 2, 1, 1, 1,
       2, 3, 4, 0, 2, 4, 2, 4])), (array([[44.18492017, 16.98881789],
       [30.06722465, 40.41123947],
       [38.18858711, 20.81726128],
       [59.22766628, 29.19796006],
       [20.0835633 , 69.81592298],
       [ 9.92511389, 25.73098987]]), array([1, 1, 4, 1, 3, 5, 1, 2, 4, 0, 2, 1, 4, 2, 4, 5, 2, 3, 2, 4, 5, 1,
       4, 3, 1, 5, 2, 2, 5, 3, 3, 1, 4, 5, 2, 1, 3, 1, 3, 4, 4, 2, 1, 5,
       0, 1, 1, 1, 4, 3, 3, 4, 1, 4, 1, 5, 1, 5, 1, 1, 5, 5, 3, 5, 3, 3,
       1, 2, 4, 3, 4, 2, 1, 1, 5, 1, 4, 5, 1, 4, 3, 1, 3, 4, 5, 2, 5, 1,
       3, 4, 1, 5, 3, 1, 1, 4, 5, 4, 2, 3, 1, 5, 3, 2, 2, 1, 5, 2, 3, 2,
       5, 5, 2, 4, 2, 5, 5, 4, 5, 5, 0, 2, 1, 4, 1, 4, 4, 3, 4, 5, 1, 5,
       3, 4, 3, 5, 1, 0, 1, 3, 1, 3, 4, 3, 4, 4, 3, 5, 5, 4, 3, 5, 3, 0,
       4, 4, 4, 3, 3, 0, 5, 5, 1, 0, 2, 3, 2, 5, 4, 4, 4, 4, 1, 0, 3, 4,
       5, 3, 3, 5, 2, 2, 5, 4, 5, 4, 2, 2, 4, 4, 3, 0, 3, 1, 0, 2, 3, 3,
       5, 1, 3, 2, 2, 1, 3, 4, 3, 5, 1, 3, 1, 5, 1, 5, 5, 1, 1, 5, 4, 1,
       4, 2, 2, 5, 4, 2, 3, 2, 3, 4, 4, 0, 2, 3, 0, 3, 2, 1, 5, 3, 3, 3,
       5, 4, 2, 1, 5, 2, 5, 2])), (array([[26.23935735, 39.56231098],
       [33.86397001, 36.21416257],
       [32.10246392, 43.52452575],
       [39.78587939, 19.72783208],
       [59.22766628, 29.19796006],
       [20.0835633 , 69.81592298],
       [ 9.92511389, 25.73098987]]), array([0, 1, 5, 2, 4, 6, 1, 3, 5, 3, 3, 2, 5, 3, 5, 6, 3, 4, 3, 5, 6, 0,
       5, 4, 1, 6, 3, 3, 6, 4, 4, 0, 5, 6, 3, 2, 4, 1, 4, 5, 5, 3, 2, 6,
       3, 0, 0, 1, 5, 4, 4, 5, 2, 5, 0, 6, 2, 6, 2, 1, 6, 6, 4, 6, 4, 4,
       0, 3, 5, 4, 5, 3, 0, 2, 6, 1, 5, 6, 2, 5, 4, 2, 4, 5, 6, 3, 6, 0,
       4, 5, 0, 6, 4, 2, 0, 5, 6, 5, 3, 4, 2, 6, 4, 3, 3, 0, 6, 3, 4, 3,
       6, 6, 3, 5, 3, 6, 6, 5, 6, 6, 3, 3, 0, 5, 1, 5, 5, 4, 5, 6, 0, 6,
       4, 5, 4, 6, 0, 3, 0, 4, 2, 4, 5, 4, 5, 5, 4, 6, 6, 5, 4, 6, 4, 3,
       5, 5, 5, 4, 4, 3, 6, 6, 2, 3, 3, 4, 3, 6, 5, 5, 5, 5, 0, 3, 4, 5,
       6, 4, 4, 6, 3, 3, 6, 5, 6, 5, 1, 3, 5, 5, 4, 3, 4, 2, 3, 3, 4, 4,
       6, 2, 4, 3, 3, 1, 4, 5, 4, 6, 1, 4, 0, 6, 0, 6, 6, 2, 2, 6, 5, 0,
       5, 3, 3, 6, 5, 3, 4, 3, 4, 5, 5, 3, 3, 4, 3, 4, 3, 0, 6, 4, 4, 4,
       6, 5, 3, 1, 6, 3, 6, 3])), (array([[46.45917139, 23.16813158],
       [17.46350686, 68.10438494],
       [59.60504708, 29.19922401],
       [32.37545229, 35.93762723],
       [38.66481538, 19.34921236],
       [28.96634723, 42.29059653],
       [ 9.92511389, 25.73098987],
       [24.01364794, 72.38323004]]), array([5, 3, 1, 5, 0, 6, 3, 4, 7, 4, 4, 5, 1, 4, 1, 6, 4, 2, 4, 1, 6, 5,
       7, 2, 3, 6, 4, 4, 6, 2, 2, 3, 7, 6, 4, 3, 2, 3, 2, 1, 7, 4, 5, 6,
       0, 5, 5, 3, 7, 2, 2, 1, 5, 7, 5, 6, 5, 6, 5, 3, 6, 6, 2, 6, 2, 2,
       5, 4, 1, 2, 1, 4, 5, 5, 6, 3, 7, 6, 5, 1, 2, 5, 2, 7, 6, 0, 6, 5,
       2, 1, 3, 6, 2, 5, 5, 7, 6, 7, 4, 2, 5, 6, 2, 4, 0, 5, 6, 4, 2, 4,
       6, 6, 4, 1, 4, 6, 6, 7, 6, 6, 0, 4, 5, 7, 3, 7, 1, 2, 1, 6, 3, 6,
       2, 1, 2, 6, 5, 4, 5, 2, 5, 2, 7, 2, 1, 7, 2, 6, 6, 1, 2, 6, 2, 4,
       7, 1, 1, 2, 2, 0, 6, 6, 5, 0, 4, 2, 0, 6, 7, 1, 1, 7, 5, 4, 2, 7,
       6, 2, 0, 6, 4, 4, 6, 1, 6, 1, 3, 4, 7, 1, 2, 4, 2, 5, 4, 4, 2, 2,
       6, 5, 2, 4, 4, 3, 2, 1, 2, 6, 3, 2, 3, 6, 5, 6, 6, 5, 5, 6, 1, 5,
       1, 4, 4, 6, 1, 4, 2, 4, 2, 1, 1, 0, 4, 2, 4, 2, 4, 5, 6, 2, 2, 2,
       6, 1, 4, 3, 6, 4, 6, 4])), (array([[41.41465903, 21.62570842],
       [37.61417321, 17.19733029],
       [26.6124459 , 43.91604964],
       [16.74238916, 69.20086704],
       [ 9.92511389, 25.73098987],
       [24.00581119, 70.53794517],
       [31.42335817, 38.75662838],
       [55.10816891, 27.75253127],
       [61.16625328, 29.87816185]]), array([6, 6, 3, 6, 7, 4, 6, 1, 5, 0, 1, 2, 3, 1, 3, 4, 1, 7, 0, 3, 4, 2,
       3, 7, 6, 4, 1, 0, 4, 8, 8, 6, 5, 4, 1, 6, 8, 6, 7, 3, 5, 0, 2, 4,
       0, 2, 6, 6, 5, 8, 7, 3, 2, 5, 6, 4, 6, 4, 6, 6, 4, 4, 8, 4, 8, 8,
       2, 1, 3, 8, 3, 0, 6, 6, 4, 6, 5, 4, 6, 3, 7, 6, 8, 5, 4, 0, 4, 2,
       7, 3, 6, 4, 7, 6, 6, 5, 4, 5, 0, 8, 6, 4, 8, 0, 0, 6, 4, 0, 8, 1,
       4, 4, 0, 3, 0, 4, 4, 5, 4, 4, 0, 0, 2, 5, 6, 5, 3, 8, 3, 4, 6, 4,
       8, 3, 7, 4, 2, 0, 2, 7, 2, 8, 5, 8, 3, 5, 8, 4, 4, 3, 7, 4, 7, 0,
       5, 3, 5, 8, 7, 0, 4, 4, 6, 0, 1, 8, 0, 4, 5, 3, 3, 5, 6, 1, 8, 5,
       4, 8, 7, 4, 1, 0, 4, 5, 4, 3, 6, 0, 5, 3, 8, 1, 8, 6, 1, 1, 8, 8,
       4, 6, 8, 1, 0, 6, 8, 3, 8, 4, 6, 8, 6, 4, 6, 4, 4, 6, 2, 4, 5, 2,
       3, 1, 0, 4, 3, 0, 7, 1, 8, 5, 3, 0, 1, 8, 1, 8, 0, 2, 4, 7, 8, 8,
       4, 3, 0, 6, 4, 1, 4, 1])), (array([[22.16577861, 70.20228452],
       [26.6124459 , 43.91604964],
       [19.74208872, 64.1000882 ],
       [39.78587939, 19.72783208],
       [15.57193046, 70.58516939],
       [60.58250295, 27.97262624],
       [25.38793378, 73.72176884],
       [ 9.92511389, 25.73098987],
       [55.37159269, 32.68544864],
       [31.42335817, 38.75662838]]), array([9, 9, 4, 9, 8, 7, 9, 3, 0, 3, 3, 1, 4, 3, 0, 7, 3, 8, 3, 0, 7, 1,
       4, 5, 9, 7, 3, 3, 7, 5, 5, 9, 0, 7, 3, 9, 5, 9, 5, 2, 0, 3, 1, 7,
       3, 1, 9, 9, 6, 5, 5, 4, 1, 6, 9, 7, 9, 7, 9, 9, 7, 7, 8, 7, 5, 5,
       1, 3, 2, 5, 0, 3, 9, 9, 7, 9, 0, 7, 9, 2, 8, 9, 5, 6, 7, 3, 7, 1,
       5, 4, 9, 7, 8, 9, 9, 0, 7, 6, 3, 5, 9, 7, 5, 3, 3, 9, 7, 3, 5, 3,
       7, 7, 3, 4, 3, 7, 7, 6, 7, 7, 3, 3, 1, 0, 9, 0, 4, 8, 4, 7, 9, 7,
       5, 4, 5, 7, 1, 3, 1, 8, 1, 8, 6, 5, 4, 0, 8, 7, 7, 4, 5, 7, 8, 3,
       6, 2, 2, 5, 5, 3, 7, 7, 9, 3, 3, 5, 3, 7, 6, 4, 2, 6, 9, 3, 5, 0,
       7, 5, 8, 7, 3, 3, 7, 0, 7, 4, 9, 3, 6, 2, 5, 3, 8, 9, 3, 3, 5, 8,
       7, 9, 5, 3, 3, 9, 5, 4, 5, 7, 9, 5, 9, 7, 9, 7, 7, 9, 1, 7, 2, 1,
       4, 3, 3, 7, 4, 3, 5, 3, 5, 2, 4, 3, 3, 5, 3, 5, 3, 1, 7, 5, 5, 5,
       7, 2, 3, 9, 7, 3, 7, 3]))]
[0.0, 75832.02223, 123143.46797, 134769.30424, 150059.18397, 150520.77443, 150997.04159, 151765.54636, 152083.78065, 152200.29395]

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/8/4cd46d5d9463dad96a37.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210609%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210609T221402Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=087f06e7fc1111e80845487707712965e4c12de5617c1730111a37ae3bf651b7)

### 4. Initialize GMM
```
yosri@ubuntu-xenial:0x01-clustering$ cat 4-main.py
#!/usr/bin/env python3

import numpy as np
initialize = __import__('4-initialize').initialize

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    pi, m, S = initialize(X, 4)
    print(pi)
    print(m)
    print(S)
yosri@ubuntu-xenial:0x01-clustering$ ./4-main.py
[0.25 0.25 0.25 0.25]
[[54.73711515 31.81393242]
 [16.84012557 31.20248225]
 [21.43215816 65.50449077]
 [32.3301925  41.80664127]]
[[[1. 0.]
  [0. 1.]]

 [[1. 0.]
  [0. 1.]]

 [[1. 0.]
  [0. 1.]]

 [[1. 0.]
  [0. 1.]]]
yosri@ubuntu-xenial:0x01-clustering$

```

### 5. PDF
```
yosri@ubuntu-xenial:0x01-clustering$ cat 5-main.py
#!/usr/bin/env python3

import numpy as np
pdf = __import__('5-pdf').pdf

if __name__ == '__main__':
    np.random.seed(0)
    m = np.array([12, 30, 10])
    S = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
    X = np.random.multivariate_normal(m, S, 10000)
    P = pdf(X, m, S)
    print(P)
yosri@ubuntu-xenial:0x01-clustering$ ./5-main.py
[3.47450910e-05 2.53649178e-06 1.80348301e-04 ... 1.24604061e-04
 1.86345129e-04 2.59397003e-05]
yosri@ubuntu-xenial:0x01-clustering$

```

### 6. Expectation
```
yosri@ubuntu-xenial:0x01-clustering$ cat 6-main.py
#!/usr/bin/env python3

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    pi, m, S = initialize(X, 4)
    g, l = expectation(X, pi, m, S)
    print(g)
    print(np.sum(g, axis=0))
        print(l)
yosri@ubuntu-xenial:0x01-clustering$ ./6-main.py
[[1.98542668e-055 1.00000000e+000 1.56526421e-185 ... 1.00000000e+000
  3.70567311e-236 1.91892348e-012]
 [6.97883333e-085 2.28658376e-279 9.28518983e-065 ... 8.12227631e-287
  1.53690661e-032 3.17417182e-181]
 [9.79811365e-234 2.28658376e-279 2.35073465e-095 ... 1.65904890e-298
  9.62514613e-068 5.67072057e-183]
 [1.00000000e+000 7.21133039e-186 1.00000000e+000 ... 2.42138447e-125
  1.00000000e+000 1.00000000e+000]]
[1. 1. 1. ... 1. 1. 1.]
-652797.7866541843
yosri@ubuntu-xenial:0x01-clustering$

```

### 7. Maximization
```
yosri@ubuntu-xenial:0x01-clustering$ cat 7-main.py
#!/usr/bin/env python3

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    pi, m, S = initialize(X, 4)
    g, _ = expectation(X, pi, m, S)
    pi, m, S = maximization(X, g)
    print(pi)
    print(m)
    print(S)
yosri@ubuntu-xenial:0x01-clustering$ ./7-main.py
[0.10104901 0.24748822 0.1193333  0.53212947]
[[54.7440558  31.80888393]
 [16.84099873 31.20560148]
 [21.42588061 65.51441875]
 [32.33208369 41.80830251]]
[[[64.05063663 -2.13941814]
  [-2.13941814 41.90354928]]

 [[72.72404579  9.96322554]
  [ 9.96322554 53.05035303]]

 [[46.20933259  1.08979413]
  [ 1.08979413 66.9841323 ]]

 [[35.04054823 -0.94790014]
  [-0.94790014 45.14948772]]]
yosri@ubuntu-xenial:0x01-clustering$

```

### 8. EM
```
yosri@ubuntu-xenial:0x01-clustering$ cat 8-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    k = 4
    pi, m, S, g, l = expectation_maximization(X, k, 150, verbose=True)
    clss = np.sum(g * np.arange(k).reshape(k, 1), axis=0)
    plt.scatter(X[:, 0], X[:, 1], s=20, c=clss)
    plt.scatter(m[:, 0], m[:, 1], s=50, c=np.arange(k), marker='*')
    plt.show()
    print(X.shape[0] * pi)
    print(m)
    print(S)
    print(l)
yosri@ubuntu-xenial:0x01-clustering$ ./8-main.py
Log Likelihood after 0 iterations: -652797.78665
Log Likelihood after 10 iterations: -94855.45662
Log Likelihood after 20 iterations: -94714.52057
Log Likelihood after 30 iterations: -94590.87362
Log Likelihood after 40 iterations: -94440.40559
Log Likelihood after 50 iterations: -94439.93891
Log Likelihood after 52 iterations: -94439.93889

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/11/9a6a49a5317fa046854f.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210609%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210609T221402Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=a6362642015e4437a0254528e29275a6e6df371b11967b78ee290c4600996597)

```
[ 761.03239903  747.62391034 1005.60275934 9985.74093129]
[[60.18888335 30.19707607]
 [ 5.05794926 24.92588821]
 [20.03438453 69.84721009]
 [29.89607379 40.12519148]]
[[[16.85183426  0.2547388 ]
  [ 0.2547388  16.49432111]]

 [[15.19555672  9.62661086]
  [ 9.62661086 15.47295413]]

 [[35.58332494 11.08419454]
  [11.08419454 33.09463207]]

 [[74.52083678  5.20755533]
  [ 5.20755533 73.87299705]]]
-94439.93889004056
yosri@ubuntu-xenial:0x01-clustering$

```

### 9. BIC
```
yosri@ubuntu-xenial:0x01-clustering$ cat 9-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
BIC = __import__('9-BIC').BIC

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    best_k, best_result, l, b = BIC(X, kmin=1, kmax=10)
    print(best_k)
    print(best_result)
    print(l)
    print(b)
    x = np.arange(1, 11)
    plt.plot(x, l, 'r')
    plt.xlabel('Clusters')
    plt.ylabel('Log Likelihood')
    plt.tight_layout()
    plt.show()
    plt.plot(x, b, 'b')
    plt.xlabel('Clusters')
    plt.ylabel('BIC')
    plt.tight_layout()
    plt.show()
yosri@ubuntu-xenial:0x01-clustering$ ./9-main.py
4
(array([0.79885962, 0.08044842, 0.06088258, 0.05980938]), array([[29.89606417, 40.12518027],
       [20.0343883 , 69.84718588],
       [60.18888407, 30.19707372],
       [ 5.05788987, 24.92583792]]), array([[[74.52101284,  5.20770764],
        [ 5.20770764, 73.8729309 ]],

       [[35.58334497, 11.08416742],
        [11.08416742, 33.09483747]],

       [[16.85183256,  0.25475122],
        [ 0.25475122, 16.4943092 ]],

       [[15.19520213,  9.62633552],
        [ 9.62633552, 15.47268905]]]))
[-98801.40298366 -96729.95558846 -95798.40406023 -94439.93888882
 -94435.87750008 -94428.62217176 -94426.71159745 -94425.5860871
 -94421.41864281 -94416.43390835]
[197649.97338694 193563.67950008 191757.17734716 189096.84790787
 189145.32603394 189187.41628084 189240.19603576 189294.54591859
 189342.81193356 189389.44336818]

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/8/3ce66d375c09b61bf6b4.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210609%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210609T221402Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4790e489313ebbf97d46fa6868937adeafe0d52806eff6b34f8cef3e7f145bd5)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/8/7434fa918f1cff45a2f9.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210609%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210609T221402Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=d08179e5ca334c4c435e04c3e9e953f45661d3a3f5723ae82417774f61fe1c5c)


### 10. Hello, sklearn!
```
yosri@ubuntu-xenial:0x01-clustering$ cat 10-main.py
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
kmeans = __import__('10-kmeans').kmeans

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    C, clss = kmeans(X, 5)
    print(C)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.scatter(C[:, 0], C[:, 1], s=50, marker='*', c=list(range(5)))
    plt.show()
yosri@ubuntu-xenial:0x01-clustering$ ./10-main.py
[[ 9.92511389 25.73098987]
 [30.06722465 40.41123947]
 [39.62770705 19.89843487]
 [59.22766628 29.19796006]
 [20.0835633  69.81592298]]

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/8/9ed61d0ec495095f80bf.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210609%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210609T221402Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=d318ec7a9cb95ab64fa40cdf73cf4e0ce3611ebb9d4753912cdc334ef0632981)

### 11. GMM
```
yosri@ubuntu-xenial:0x01-clustering$ cat 11-main.py
#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
gmm = __import__('11-gmm').gmm

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)

    pi, m, S, clss, bic = gmm(X, 4)
    print(pi)
    print(m)
    print(S)
    print(bic)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.scatter(m[:, 0], m[:, 1], s=50, marker='*', c=list(range(4)))
    plt.show()
yosri@ubuntu-xenial:0x01-clustering$ ./11-main.py
[0.06054777 0.70467829 0.15613693 0.07863701]
[[60.23478926 30.22428892]
 [30.59365755 40.63753599]
 [17.17946347 32.21805142]
 [20.00762691 70.0169833 ]]
[[[ 1.65286199e+01  1.02543709e-01]
  [ 1.02543709e-01  1.65687019e+01]]

 [[ 7.09781098e+01 -2.18081013e+00]
  [-2.18081013e+00  7.78058769e+01]]

 [[ 1.41559799e+02  7.97703830e+01]
  [ 7.97703830e+01  6.39836725e+01]]

 [[ 3.54632094e+01  1.12791052e+01]
  [ 1.12791052e+01  3.21044161e+01]]]
189727.91411998263

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/8/4d20ec068e424c587a98.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210609%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210609T221402Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2e0c480622233be45fc42a1017d5b0b7500fa93802dc28cab01418cd230dbe34)

### 12. Agglomerative
```
yosri@ubuntu-xenial:0x01-clustering$ cat 12-main.py
#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
agglomerative = __import__('12-agglomerative').agglomerative

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=100)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=100)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    clss = agglomerative(X, 100)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.show()
yosri@ubuntu-xenial:0x01-clustering$ ./12-main.py
```
> Copyright © 2021 Holberton School. All rights reserved.

![Yosri Ghorbel](https://pbs.twimg.com/media/E3YEO7kXwAU9x6x?format=png&name=4096x4096)
