import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as pl
general_list = []

list_allocated = []
list_max_allocated = []
list_reserved = []
list_max_reserved = []
list_cached = []
list_max_cached = []

with open('../Memory_allocated_first.txt') as mem_file:
    general_list.append(mem_file.readlines())

general_list = [int(i) for i in general_list[0]]

for i in range(0, len(general_list), 6):
    list_allocated.append(general_list[i])
    list_reserved.append(general_list[i+1])
    list_cached.append(general_list[i+2])

    list_max_allocated.append(general_list[i+3])

    list_max_reserved.append(general_list[i+4])
    list_max_cached.append(general_list[i+5])


    # while True:
    #     if not mem_file.readline():
    #         break
    #     list_allocated.append(int(mem_file.readline()))
    #     list_reserved.append(int(mem_file.readline()))
    #     list_cached.append(int(mem_file.readline()))
    #
    #     list_max_allocated.append(int(mem_file.readline()))
    #
    #     list_max_reserved.append((mem_file.readline()))
    #     list_max_cached.append((mem_file.readline()))

x = np.array(range(len(list_allocated)))

fig1 = plt.figure(figsize=(15,8))
plt.plot(x, list_allocated, label='allocated')
plt.plot(x, list_reserved, label='reserved')
plt.plot(x, list_cached, label='cached')
plt.plot(x, list_max_allocated, label='max_alloc')
plt.plot(x, list_max_reserved, label='max_reserved')
plt.plot(x, list_max_cached, label='max_cached')

# list_allocated = list_allocated.sort()
# list_allocated = list_allocated.sort()
# plt.plot(x, list_allocated)
fig1.legend()

plt.show()




