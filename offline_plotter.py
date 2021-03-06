from dynamic_plotter import *
import numpy as np

d1 = DynamicPlot(window_x=100, title='Predictions', xlabel='time_step', ylabel='value',legend = True)
d2 = DynamicPlot(window_x=100, title='Cumulants', xlabel='time_step', ylabel='value')
d3 = DynamicPlot(window_x=100, title='RUPEE', xlabel='time_step', ylabel='value')
d4 = DynamicPlot(window_x=100, title='UDE', xlabel='time_step', ylabel='value')
d5 = DynamicPlot(window_x=100, title='TD error', xlabel='time_step', ylabel='value')

d1.add_line('GVF1')
d1.add_line('GVF2')
d1.add_line('GVF3')
d1.add_line('GVF4')
d1.add_line('GVF5')
d1.add_line('GVF6')
d1.add_line('GVF7')
d1.add_line('GVF8')
d1.add_line('GVF9')
d1.add_line('GVF10')

d2.add_line('GVF1 Cumulant')
d2.add_line('GVF2 Cumulant')
d2.add_line('GVF3 Cumulant')
d2.add_line('GVF4 Cumulant')
d2.add_line('GVF5 Cumulant')
d2.add_line('GVF6 Cumulant')
d2.add_line('GVF7 Cumulant')
d2.add_line('GVF8 Cumulant')
d2.add_line('GVF9 Cumulant')
d2.add_line('GVF10 Cumulant')

d3.add_line('GVF1 RUPEE')
d3.add_line('GVF2 RUPEE')
d3.add_line('GVF3 RUPEE')
d3.add_line('GVF4 RUPEE')
d3.add_line('GVF5 RUPEE')
d3.add_line('GVF6 RUPEE')
d3.add_line('GVF7 RUPEE')
d3.add_line('GVF8 RUPEE')
d3.add_line('GVF9 RUPEE')
d3.add_line('GVF10 RUPEE')

d4.add_line('GVF1 UDE')
d4.add_line('GVF2 UDE')
d4.add_line('GVF3 UDE')
d4.add_line('GVF4 UDE')
d4.add_line('GVF5 UDE')
d4.add_line('GVF6 UDE')
d4.add_line('GVF7 UDE')
d4.add_line('GVF8 UDE')
d4.add_line('GVF9 UDE')
d4.add_line('GVF10 UDE')

d5.add_line('GVF1 TD error')
d5.add_line('GVF2 TD error')
d5.add_line('GVF3 TD error')
d5.add_line('GVF4 TD error')
d5.add_line('GVF5 TD error')
d5.add_line('GVF6 TD error')
d5.add_line('GVF7 TD error')
d5.add_line('GVF8 TD error')
d5.add_line('GVF9 TD error')
d5.add_line('GVF10 TD error')


npspreds = np.loadtxt('predictions.txt')
scumul = np.loadtxt('cumulants.txt')
srp = np.loadtxt('RUPEE.txt')
sud = np.loadtxt('UDE.txt')
stde = np.loadtxt('TDERROR.txt')

for i in range(npspreds.shape[0]):
    d1.update(i,np.array(npspreds[i].tolist()))
    d2.update(i,np.array(scumul[i].tolist()))
    d3.update(i,np.array(srp[i].tolist()))
    d4.update(i,np.array(sud[i].tolist()))
    d5.update(i,np.array(stde[i].tolist()))
    # change to make fast and slow plotting
    time.sleep(0.0)
while True:
    pass