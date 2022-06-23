# import matplotlib.pyplot as plt
# import numpy as np

# data = { 1: 13428, 2: 11706, 3: 9277, 4: 7320, 5: 6222, 6: 4868, 7: 4045, 8: 3472, 9: 2976, 10: 27627}
# # data = {1: 13428, 2: 11706, 3: 9277, 4: 7320, 5: 6222, 6: 4868, 7: 4045, 8: 3472, 9: 2976, 10: 13814, 0: 13813}
# # data = { 1: 13428, 2: 11706, 3: 9277, 4: 7320, 5: 6222, 6: 4868, 7: 4045, 8: 3472, 9: 2976, 10: 2599}
# # data = {1: 13428, 2: 11706, 3: 9277, 4: 7320, 5: 6222, 6: 4868, 7: 4045, 8: 3472, 9: 2976, 10: 2599,  11: 2203, 12: 1937, 13: 1656, 14: 1434, 15: 1289, 16: 1111, 17: 1030, 18: 948, 19: 836, 20: 807, 21: 717, 22: 556, 23: 519, 24: 525, 25: 483, 26: 409, 27: 390, 28: 370, 29: 329, 30: 261, 31: 300, 32: 273,  33: 261, 34: 231, 35: 200, 36: 226,  37: 196, 38: 186,39: 200, 40: 161, 41: 167, 42: 165,  43: 138, 44: 135, 45: 142,  48: 112, 46: 106, 47: 105, 51: 104, 49: 91, 55: 90, 50: 89, 52: 81, 53: 79, 54: 78, 58: 73, 56: 71, 61: 69, 67: 68, 57: 68, 72: 63, 60: 63, 62: 62, 59: 55, 65: 54, 63: 52, 66: 51, 64: 49, 69: 47, 70: 44, 71: 43, 68: 42, 77: 40, 86: 39, 87: 38, 74: 36, 75: 35, 82: 35, 81: 35, 84: 33, 73: 32, 80: 31, 76: 31, 94: 29, 98: 29, 83: 29, 95: 27, 79: 27, 85: 27, 91: 27, 99: 26, 88: 25, 92: 24, 89: 24, 90: 23, 93: 22, 78: 22, 97: 21, 96: 20  , 100: 1541}
# data = dict(sorted(data.items()))
# print(data)
# names = list(data.keys())
# values = list(data.values())

# plt.ylim([0,30000])
# plt.bar(range(len(data)), values, tick_label=names)
# plt.savefig('foo.png')
# plt.show()
# print()
# data_list = [(k, v) for k, v in data.items()]
# sss=''
# for item in data_list:
#     sss += str(item) + ' '
# print(sss)

import matplotlib.pyplot as plt
import numpy as np




x=[[27627, 13814], [2976,1488],
   [3472, 1736], [4045, 2023]]
y=[[2.15803,1.06633], [0.21174,0.1101],
   [0.21827,0.1099],[0.2241,0.11037]]



for i in range(4):
    plt.plot(x[i], y[i], label = "in-degree "+str(10-i))
    # plt.plot(y, x, label = "line 2")
# plt.plot(x, np.sin(x), label = "curve 1")
# plt.plot(x, np.cos(x), label = "curve 2")
plt.legend()
plt.savefig('test.png')
plt.show()
