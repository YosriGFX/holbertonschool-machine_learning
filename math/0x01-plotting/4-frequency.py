#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='k')
plt.xticks(ticks=range(0, 101, 10))
plt.xlabel('Grades')
plt.xlim(0, 100)
plt.ylim(0, 30)
plt.ylabel('Number of Students')
plt.title('Project A')
plt.show()
