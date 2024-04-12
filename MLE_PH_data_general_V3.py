import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib import collections as mc
import pylab as pl
from matplotlib.ticker import FormatStrFormatter

np.set_printoptions(precision=4, linewidth=100, suppress=True)

# Import data

training_data = pd.read_csv("PH_training_data_setp.csv", usecols=range(1, 29))
test_data = pd.read_csv("PH_test_data_setp.csv", usecols=range(1, 29))

n = len(training_data)

# Training data

T_Out_0_tr = list(training_data['outdoorT'])

V_Sup_0_tr = list(training_data['SAV'])

T_Stp_0_A13_tr = list(training_data['RMA13.RTS'])
T_Stp_0_A15_tr = list(training_data['RMA15.RTS'])
T_Stp_0_A17_tr = list(training_data['RMA17.RTS'])

T_In_0_A13_tr = list(training_data['RMA13.RT'])
T_In_0_A15_tr = list(training_data['RMA15.RT'])
T_In_0_A17_tr = list(training_data['RMA17.RT'])

T_Sup_0_A13_tr = list(training_data['RMA13.DT'])
T_Sup_0_A15_tr = list(training_data['RMA15.DT'])
T_Sup_0_A17_tr = list(training_data['RMA17.DT'])

T_In_1s_A13_tr = list(training_data['RMA13.1s'])
T_In_10s_A13_tr = list(training_data['RMA13.10s'])
T_In_100s_A13_tr = list(training_data['RMA13.100s'])
T_In_1000s_A13_tr = list(training_data['RMA13.1000s'])

T_In_1s_A15_tr = list(training_data['RMA15.1s'])
T_In_10s_A15_tr = list(training_data['RMA15.10s'])
T_In_100s_A15_tr = list(training_data['RMA15.100s'])
T_In_1000s_A15_tr = list(training_data['RMA15.1000s'])

T_In_1s_A17_tr = list(training_data['RMA17.1s'])
T_In_10s_A17_tr = list(training_data['RMA17.10s'])
T_In_100s_A17_tr = list(training_data['RMA17.100s'])
T_In_1000s_A17_tr = list(training_data['RMA17.1000s'])

# Test data

T_Out_0_test = list(test_data['outdoorT'])

T_Stp_0_A13_test = list(test_data['RMA13.RTS'])
T_Stp_0_A15_test = list(test_data['RMA15.RTS'])
T_Stp_0_A17_test = list(test_data['RMA17.RTS'])

V_Sup_0_test = list(test_data['SAV'])

T_In_0_A13_test = list(test_data['RMA13.RT'])
T_In_0_A15_test = list(test_data['RMA15.RT'])
T_In_0_A17_test = list(test_data['RMA17.RT'])

T_Sup_0_A13_test = list(test_data['RMA13.DT'])
T_Sup_0_A15_test = list(test_data['RMA15.DT'])
T_Sup_0_A17_test = list(test_data['RMA17.DT'])

T_In_1s_A13_test = list(test_data['RMA13.1s'])
T_In_10s_A13_test = list(test_data['RMA13.10s'])
T_In_100s_A13_test = list(test_data['RMA13.100s'])
T_In_1000s_A13_test = list(test_data['RMA13.1000s'])

T_In_1s_A15_test = list(test_data['RMA15.1s'])
T_In_10s_A15_test = list(test_data['RMA15.10s'])
T_In_100s_A15_test = list(test_data['RMA15.100s'])
T_In_1000s_A15_test = list(test_data['RMA15.1000s'])

T_In_1s_A17_test = list(test_data['RMA17.1s'])
T_In_10s_A17_test = list(test_data['RMA17.10s'])
T_In_100s_A17_test = list(test_data['RMA17.100s'])
T_In_1000s_A17_test = list(test_data['RMA17.1000s'])

# PRIOR ESTIMATION (A13/1000s case)
# Estimate the mean vectors
variables = ["1s_A13", "10s_A13", "100s_A13", "1000s_A13",
             "1s_A15", "10s_A15", "100s_A15", "1000s_A15",
             "1s_A17", "10s_A17", "100s_A17", "1000s_A17"]

for variable in variables:
    globals()['mean_est_' + variable] = np.array([[np.mean(globals()['T_In_' + variable + '_tr'])],
                                                  [np.mean(T_Out_0_tr)],
                                                  [np.mean(globals()['T_Stp_0_' + variable[-3:] + '_tr'])],
                                                  [np.mean(globals()['T_In_0_' + variable[-3:] + '_tr'])],
                                                  [np.mean(globals()['T_Sup_0_' + variable[-3:] + '_tr'])],
                                                  [np.mean(V_Sup_0_tr)]])
    # print('mean_est_' + variable + ':')
    # print(globals()['mean_est_' + variable])

# Estimate the covariance matrix
for variable in variables:

    Sn = np.zeros((6, 6))

    for i in range(0, n):
        z = np.array([[globals()['T_In_' + variable + '_tr'][i]],
                      [T_Out_0_tr[i]],
                      [globals()['T_Stp_0_' + variable[-3:] + '_tr'][i]],
                      [globals()['T_In_0_' + variable[-3:] + '_tr'][i]],
                      [globals()['T_Sup_0_' + variable[-3:] + '_tr'][i]],
                      [V_Sup_0_tr[i]]])

        dot_prod = np.matmul((z - globals()['mean_est_' + variable]),
                             (np.transpose(z - globals()['mean_est_' + variable])))
        Sn = Sn + dot_prod

    globals()['Cov_est_' + variable] = np.array(Sn / n)

    # print('Cov_est_' + variable + ':')
    # print(globals()['Cov_est_' + variable])

# Partition
for variable in variables:
    globals()['sig_a_' + variable] = globals()['Cov_est_' + variable][0, 0]
    globals()['sig_ab_' + variable] = globals()['Cov_est_' + variable][0, 1:6]
    globals()['sig_ab_t_' + variable] = np.transpose([globals()['Cov_est_' + variable][0, 1:6]])
    globals()['sig_b_' + variable] = globals()['Cov_est_' + variable][1:, 1:]

    globals()['mean_a_' + variable] = globals()['mean_est_' + variable][0]
    globals()['mean_b_' + variable] = globals()['mean_est_' + variable][1:]

    # print(globals()['sig_b_' + variable])

# Sensor data (evidences)
T_Out_0_sensor = 20

T_In_0_A13_sensor = 21
T_Stp_0_A13_control = 20.5
T_Sup_0_A13_sensor = 20
V_Sup_0_A13_sensor = 0.02
x_b_A13 = np.array([[T_Out_0_sensor],
                    [T_Stp_0_A13_control],
                    [T_In_0_A13_sensor],
                    [T_Sup_0_A13_sensor],
                    [V_Sup_0_A13_sensor]])

T_In_0_A15_sensor = 23
T_Stp_0_A15_control = 26.5
T_Sup_0_A15_sensor = 35
V_Sup_0_A15_sensor = 0.02
x_b_A15 = np.array([[T_Out_0_sensor],
                    [T_Stp_0_A15_control],
                    [T_In_0_A15_sensor],
                    [T_Sup_0_A15_sensor],
                    [V_Sup_0_A15_sensor]])

T_In_0_A17_sensor = 21
T_Stp_0_A17_control = 20.5
T_Sup_0_A17_sensor = 20
V_Sup_0_A17_sensor = 0.02
x_b_A17 = np.array([[T_Out_0_sensor],
                    [T_Stp_0_A17_control],
                    [T_In_0_A17_sensor],
                    [T_Sup_0_A17_sensor],
                    [V_Sup_0_A17_sensor]])

# Posterior estimation
for variable in variables:
    globals()['var_a_giv_b_' + variable] = globals()['sig_a_' + variable] \
                                           - np.matmul(globals()['sig_ab_' + variable],
                                                       np.matmul(np.linalg.inv(
                                                           globals()['sig_b_' + variable]),
                                                           globals()['sig_ab_t_' + variable]))

    # print(np.matmul(np.linalg.inv(globals()['sig_b_' + variable]), globals()['sig_ab_t_' + variable]))
    globals()['sig_a_giv_b_' + variable] = math.sqrt(globals()['var_a_giv_b_' + variable])
    globals()['mean_a_giv_b_' + variable] = globals()['mean_a_' + variable] \
                                            + np.matmul(globals()['sig_ab_' + variable],
                                                        np.matmul(np.linalg.inv(globals()['sig_b_' + variable]),
                                                                  (globals()['x_b_' + variable[-3:]]
                                                                   - globals()['mean_b_' + variable])))
    # print('sig_a_giv_b_' + variable + ':')
    # print(globals()['sig_a_giv_b_' + variable])
    # print(['mean_a_giv_b_' + variable + ':'])
    # print(globals()['mean_a_giv_b_' + variable])

# Define 95% CI
for variable in variables:
    globals()['lo_' + variable] = globals()['mean_a_giv_b_' + variable] \
                                  - 2 * (globals()['sig_a_giv_b_' + variable] / math.sqrt(len(training_data)))
    globals()['up_' + variable] = globals()['mean_a_giv_b_' + variable] \
                                  + 2 * (globals()['sig_a_giv_b_' + variable] / math.sqrt(len(training_data)))
    # print(globals()['lo_' + variable])
    # print(globals()['mean_a_giv_b_' + variable])
    # print(globals()['up_' + variable])
    # print('----')

# Find time to failure: Slope=(Y2-Y1/X2-X1); Intercept=(Y1-b*X1)

for room in ['A13', 'A15', 'A17']:
    globals()['slope_' + room + '_1'] = (globals()['mean_a_giv_b_10s_' + room]
                                         - globals()['mean_a_giv_b_1s_' + room])/9
    globals()['intercept_' + room + '_1'] = globals()['mean_a_giv_b_10s_' + room]\
                                            - globals()['slope_' + room + '_1'] * 10

    globals()['slope_' + room + '_2'] = (globals()['mean_a_giv_b_100s_' + room]
                                         - globals()['mean_a_giv_b_10s_' + room])/90
    globals()['intercept_' + room + '_2'] = globals()['mean_a_giv_b_100s_' + room]\
                                            - globals()['slope_' + room + '_2'] * 100

    globals()['slope_' + room + '_3'] = (globals()['mean_a_giv_b_1000s_' + room]
                                         - globals()['mean_a_giv_b_100s_' + room])/900
    globals()['intercept_' + room + '_3'] = globals()['mean_a_giv_b_1000s_' + room]\
                                            - globals()['slope_' + room + '_3'] * 1000
    # print('slope_1 ')
    # print(globals()['slope_' + room + '_1'])
    # print('intercept_1 ')
    # print(globals()['intercept_' + room + '_1'])
    # print('slope_2 ')
    # print(globals()['slope_' + room + '_2'])
    # print('intercept_2 ')
    # print(globals()['intercept_' + room + '_2'])
    # print('slope_3 ')
    # print(globals()['slope_' + room + '_3'])
    # print('intercept_3 ')
    # print(globals()['intercept_' + room + '_3'])

globals()['Failure_A13'] = 'NA'
globals()['Failure_A15'] = 'NA'
globals()['Failure_A17'] = 'NA'
globals()['CI_A13'] = 'NA'
globals()['CI_A15'] = 'NA'
globals()['CI_A17'] = 'NA'
Lo_Temperature = 'NA'

for room in ['A13', 'A15', 'A17']:
    for t in range(1, 1000):
        if t <= 10:
            Temperature = globals()['slope_' + room + '_1'] * t + globals()['intercept_' + room + '_1']
            if Temperature < 20 or Temperature > 25:
                globals()['Failure_' + room] = t
                Lo_Temperature = Temperature - 2 * (globals()['sig_a_giv_b_10s_' + room] / math.sqrt(len(training_data)))
                globals()['CI_' + room] = (Lo_Temperature - globals()['intercept_' + room + '_1'])\
                                          / globals()['slope_' + room + '_1']
                globals()['delta_' + room] = abs(globals()['Failure_' + room] - globals()['CI_' + room])
                break
        elif 10 < t <= 100:
            Temperature = globals()['slope_' + room + '_2'] * t + globals()['intercept_' + room + '_2']
            if Temperature < 20 or Temperature > 25:
                globals()['Failure_' + room] = t
                Lo_Temperature = Temperature - 2 * (globals()['sig_a_giv_b_100s_' + room] / math.sqrt(len(training_data)))
                globals()['CI_' + room] = (Lo_Temperature - globals()['intercept_' + room + '_2'])\
                                          / globals()['slope_' + room + '_2']
                globals()['delta_' + room] = abs(globals()['Failure_' + room] - globals()['CI_' + room])
                break
        else:
            Temperature = globals()['slope_' + room + '_3'] * t + globals()['intercept_' + room + '_3']
            if Temperature < 20 or Temperature > 25:
                globals()['Failure_' + room] = t
                Lo_Temperature = Temperature - 2 * (globals()['sig_a_giv_b_1000s_' + room] / math.sqrt(len(training_data)))
                globals()['CI_' + room] = (Lo_Temperature - globals()['intercept_' + room + '_3'])\
                                          / globals()['slope_' + room + '_3']
                globals()['delta_' + room] = abs(globals()['Failure_' + room] - globals()['CI_' + room])
                break

# print(len(training_data))
# print(Temperature)
# print(globals()['Failure_A13'])
# print(Lo_Temperature)
# print(globals()['CI_A13'])
# print(globals()['delta_A13'])

# print(globals()['Failure_A13'])
# print(globals()['Failure_A15'])
# print(globals()['Failure_A17'])

# TRAJECTORIES

# Assign points
x = [1, 10, 100, 1000]
y_A13 = [globals()['mean_a_giv_b_1s_A13'][0], globals()['mean_a_giv_b_10s_A13'][0],
         globals()['mean_a_giv_b_100s_A13'][0], globals()['mean_a_giv_b_1000s_A13'][0]]
# y_A13_lo = [globals()['lo_1s_A13'][0], globals()['lo_10s_A13'][0],
#             globals()['lo_100s_A13'][0], globals()['lo_1000s_A13'][0]]
# y_A13_up = [globals()['up_1s_A13'][0], globals()['up_10s_A13'][0],
#             globals()['up_100s_A13'][0], globals()['up_1000s_A13'][0]]

y_A15 = [globals()['mean_a_giv_b_1s_A15'][0], globals()['mean_a_giv_b_10s_A15'][0],
         globals()['mean_a_giv_b_100s_A15'][0], globals()['mean_a_giv_b_1000s_A15'][0]]
# y_A15_lo = [globals()['lo_1s_A15'][0], globals()['lo_10s_A15'][0],
#             globals()['lo_100s_A15'][0], globals()['lo_1000s_A15'][0]]
# y_A15_up = [globals()['up_1s_A15'][0], globals()['up_10s_A15'][0],
#             globals()['up_100s_A15'][0], globals()['up_1000s_A15'][0]]

y_A17 = [globals()['mean_a_giv_b_1s_A17'][0], globals()['mean_a_giv_b_10s_A17'][0],
         globals()['mean_a_giv_b_100s_A17'][0], globals()['mean_a_giv_b_1000s_A17'][0]]
# y_A17_lo = [globals()['lo_1s_A17'][0], globals()['lo_10s_A17'][0],
#             globals()['lo_100s_A17'][0], globals()['lo_1000s_A17'][0]]
# y_A17_up = [globals()['up_1s_A17'][0], globals()['up_10s_A17'][0],
#             globals()['up_100s_A17'][0], globals()['up_1000s_A17'][0]]

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex='all', sharey='all')

# Plot
# plt.xscale("log")

ax1.plot(x, y_A13)
ax1.set(ylabel='Room A13')
ax1.plot(x, [20, 20, 20, 20], color='black', linestyle='--', linewidth=0.5)
ax1.plot(x, [25, 25, 25, 25], color='black', linestyle='--', linewidth=0.5)
if globals()['Failure_A13'] != 'NA':
    ax1.axvline(globals()['Failure_A13'], color='red', linestyle='--', linewidth=0.5,
                label=('Time to failure: ' + str(int(globals()['Failure_A13']/60)) + ' min' + ' ± '
                       + str(int(globals()['delta_A13'])) + ' sec'))
    ax1.legend(loc='lower right')
# ax1.plot(x, y_A13_lo, color='black', linestyle='--', linewidth=0.5)
# ax1.plot(x, y_A13_up, color='black', linestyle='--', linewidth=0.5)

ax2.plot(x, y_A15)
ax2.set(ylabel='Room A15')
ax2.plot(x, [20, 20, 20, 20], color='black', linestyle='--', linewidth=0.5)
ax2.plot(x, [25, 25, 25, 25], color='black', linestyle='--', linewidth=0.5)
if globals()['Failure_A15'] != 'NA':
    ax2.axvline(globals()['Failure_A15'], color='red', linestyle='--', linewidth=0.5,
                label=('Time to failure: ' + str(int(globals()['Failure_A15']/60)) + ' min' + ' ± '
                       + str(int(globals()['delta_A15'])) + ' sec'))
    ax2.legend(loc='lower right')
# ax2.plot(x, y_A15_lo, color='black', linestyle='--', linewidth=0.5)
# ax2.plot(x, y_A15_up, color='black', linestyle='--', linewidth=0.5)

ax3.plot(x, y_A17)
ax3.set(ylabel='Room A17')
ax3.plot(x, [20, 20, 20, 20], color='black', linestyle='--', linewidth=0.5)
ax3.plot(x, [25, 25, 25, 25], color='black', linestyle='--', linewidth=0.5)
if globals()['Failure_A17'] != 'NA':
    ax3.axvline(globals()['Failure_A17'], color='red', linestyle='--', linewidth=0.5,
                label=('Time to failure: ' + str(int(globals()['Failure_A17'] / 60)) + ' min' + ' ± '
                       + str(int(globals()['delta_A17'])) + ' sec'))
    ax3.legend(loc='lower right')
# ax3.plot(x, y_A17_lo, color='black', linestyle='--', linewidth=0.5)
# ax3.plot(x, y_A17_up, color='black', linestyle='--', linewidth=0.5)

ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
fig.suptitle('Future inside tempartures [℃]: Trajectories')
plt.xlabel("Time [s]")
fig.text(0.01, 0.5, 'Tempartures [℃]', va='center', rotation='vertical')

# PDFs
fig2, axs = plt.subplots(3, 4, sharex='all', sharey='all')

i = 0
j = 0

for variable in variables:
    globals()['x_' + variable] = np.linspace(globals()['mean_a_giv_b_' + variable]
                                             - 3 * globals()['sig_a_giv_b_' + variable],
                                             globals()['mean_a_giv_b_' + variable]
                                             + 3 * globals()['sig_a_giv_b_' + variable], 100)

    globals()['y_' + variable] = stats.norm.pdf(globals()['x_' + variable], globals()['mean_a_giv_b_' + variable],
                                                globals()['sig_a_giv_b_' + variable])

    axs[i, j].plot(globals()['x_' + variable], globals()['y_' + variable])
    # axs[i, j].annotate(int(globals()['mean_a_giv_b_' + variable][0]) % globals()['sig_a_giv_b_' + variable],
    #                    xy=(0.8, 0.8), xycoords='axes fraction')
    axs[i, j].annotate((r'  $\mu=%.2f$' % (globals()['mean_a_giv_b_' + variable])),
                       xy=(0.25, 0.85), xycoords='axes fraction', fontsize=8)
    axs[i, j].annotate((r'  $\sigma=%.2f$' % (globals()['sig_a_giv_b_' + variable])),
                       xy=(0.25, 0.7), xycoords='axes fraction', fontsize=8)

    j = j + 1
    if j > 3:
        j = 0
        i = i + 1
axs[0, 0].set(ylabel='Room A13')
axs[1, 0].set(ylabel='Room A15')
axs[2, 0].set(ylabel='Room A17')
axs[0, 0].set(title='1s')
axs[0, 1].set(title='10s')
axs[0, 2].set(title='100s')
axs[0, 3].set(title='1000s')
fig2.suptitle('Future inside tempartures [℃]: Distributions')
fig2.text(0.5, 0.01, 'Temparture [℃]', ha='center')
fig2.text(0.01, 0.5, 'Probability', va='center', rotation='vertical')

# ACCURACY TEST: plot prediction vs ground truth for test dataset
for variable in variables:
    globals()['mean_a_giv_b_test_' + variable] = [0] * len(test_data)

for variable in variables:
    for t in range(0, len(test_data)):
        x_b_test = np.array([[T_Out_0_test[t]],
                             [globals()['T_Stp_0_' + variable[-3:] + '_test'][t]],
                             [globals()['T_In_0_' + variable[-3:] + '_test'][t]],
                             [globals()['T_Sup_0_' + variable[-3:] + '_test'][t]],
                             [V_Sup_0_test[t]]])
        globals()['mean_a_giv_b_test_' + variable][t] = globals()['mean_a_' + variable] \
            + np.matmul(globals()['sig_ab_' + variable], np.matmul(np.linalg.inv(globals()['sig_b_' + variable]),
                                                                   (globals()['x_b_test']
                                                                    - globals()['mean_b_' + variable])))

fig_test, ax_test = plt.subplots(3, 4, sharex='all', sharey='all')
i = 0
j = 0

for variable in variables:
    ax_test[i, j].scatter(globals()['T_In_' + variable + '_test'], globals()['mean_a_giv_b_test_' + variable], s=0.1)
    j = j + 1
    if j > 3:
        j = 0
        i = i + 1

plt.xlim(15, 30)
plt.ylim(15, 30)
plt.gca().set_aspect('equal', adjustable='box')
plt.suptitle("Prediction accuracy on test data")
fig_test.text(0.5, 0.01, 'Ground truth [℃]', ha='center')
fig_test.text(0.01, 0.5, 'Prediction [℃]', va='center', rotation='vertical')
ax_test[0, 0].set(ylabel='Room A13')
ax_test[1, 0].set(ylabel='Room A15')
ax_test[2, 0].set(ylabel='Room A17')
ax_test[0, 0].set(title='1s')
ax_test[0, 1].set(title='10s')
ax_test[0, 2].set(title='100s')
ax_test[0, 3].set(title='1000s')
plt.show()
