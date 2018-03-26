from utils import *
from lib_robotis_hack import *
from dynamic_plotter import *
from gvf import GVF
import signal
import time
import pickle

servo1_data = None
flag_stop = False
spreds = []
scumul = []
srp = []
sud = []
stde = []

def main():
    global servo1_data, flag_stop, spreds, scumul, srp, sud, stde
    servo1_data = []

    # servo connection step
    D = USB2Dynamixel_Device(dev_name="/dev/tty.usbserial-AI03QD8V", baudrate=1000000)
    s1 = Robotis_Servo(D, 2)
    s2 = Robotis_Servo(D, 3)
    s1.move_angle(1.5,blocking=False)

    #plotting
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

    # GVFs variables
    n_bin = last_bin
    num_state = n_bin + 1
    active_features = 1

    # bin config
    bins_off = np.linspace(0, 3, n_bin, endpoint=False)
    bins_on = np.linspace(0, 6, n_bin, endpoint=False)

    # environemnt variables
    dir = 1
    t = 0

    # last cumulant
    last_angle = None
    last_load = None
    last_temp = None
    last_voltage = None
    last_time = None
    current_time = None
    # keeping states
    last_state_on = None
    last_state_off = None
    gvfs = []

    # create all gvfs

    #0-predict next angular position
    gvfs.append(GVF(num_state))
    #1-predict next load
    gvfs.append(GVF(num_state))
    #2-predict next temp
    gvfs.append(GVF(num_state))
    #3-predict next voltage
    gvfs.append(GVF(num_state))
    #4-off policy counting steps to -1.5
    gvfs.append(GVF(num_state,is_offpolicy=True,bhv_policy=back_forth_policy,target_policy=go_to_neg_policy))
    #5-off policy how long (time) to +1.5
    gvfs.append(GVF(num_state,is_offpolicy=True,bhv_policy=back_forth_policy,target_policy=go_to_pos_policy))
    #6-on policy counting steps to -1.5
    gvfs.append(GVF(num_state))
    #7-on policy how long (time) to +1.5
    gvfs.append(GVF(num_state))
    #8-predict load of cumulative ten steps into the future with gamma 0.9
    gvfs.append(GVF(num_state))
    #9-predict angular position of cumulative ten steps into the future with gamma 0.9
    gvfs.append(GVF(num_state))
    while True:
        # real time cumulant update
        # reading data for servo 1
        [ang, position, speed, load, voltage, temperature] = read_data(s1)

        # robot policy
        dir = policy_robot(s1,ang,dir)
        # direction is the action that we are taking
        action = dir

        current_state_off = get_angle_bin_off(ang,dir,bins_off)
        current_state_on = get_angle_bin_on(ang,dir,bins_on)

        # TD lambda
        state_on = last_state_on
        state_off = last_state_off
        state_prime_on = current_state_on
        state_prime_off = current_state_off
        pred = None

        is_suprise = True
        is_pavlov = True
        if last_state_off == None:

            # set all GVFs initial state
            for i in range(len(gvfs)):
                if(gvfs[i].is_offpolicy):
                    gvfs[i].set_initial_state(current_state_off)
                else:
                    gvfs[i].set_initial_state(current_state_on)
            last_time = time.time()
        else:

            if is_suprise:
                if t > 100:
                    gvfs[0].td(last_angle*10,0,feature_vector_on(state_prime_on))
                else:
                    gvfs[0].td(last_angle,0,feature_vector_on(state_prime_on))
            # gvf update
            if not is_suprise:
                gvfs[0].td(last_angle,0,feature_vector_on(state_prime_on))
            gvfs[1].td(last_load / 10,0,feature_vector_on(state_prime_on))
            gvfs[2].td(last_temp,0,feature_vector_on(state_prime_on))
            gvfs[3].td(last_voltage,0,feature_vector_on(state_prime_on))
            gvfs[4].gtd(cummlant_negative(state_prime_off),gamma_to_neg(state_prime_off),feature_vector_off(state_prime_off),action)
            gvfs[5].gtd(((time.time()-last_time)*100)*cummlant_positive(state_prime_off),gamma_to_pos(state_prime_off),feature_vector_off(state_prime_off),action)
            gvfs[6].td(cummlant_negative(state_prime_on),gamma_to_neg(state_prime_on),feature_vector_on(state_prime_on))
            gvfs[7].td(((time.time()-last_time)*100*cummlant_negative(state_prime_on)),gamma_to_neg(state_prime_on),feature_vector_off(state_prime_on))
            gvfs[8].td(last_angle,0.9,feature_vector_on(state_prime_on))
            gvfs[9].td(last_load / 10,0.9,feature_vector_on(state_prime_on))
        # plot and save data

        preds = []
        cumul = []
        rp = []
        ud = []
        tde = []
        for i in range(len(gvfs)):
            preds.append(gvfs[i].get_prediction(state_prime_on))
            cumul.append(gvfs[i].cum)
            rp.append(gvfs[i].rupee.get_current_val())
            ud.append(gvfs[i].ude.get_current_val())
            tde.append(gvfs[i].sdelta)

        if not last_state_off == None:
            spreds.append(preds)
            scumul.append(cumul)
            srp.append(rp)
            sud.append(ud)
            stde.append(tde)

        d1.update(t, preds)
        d2.update(t, cumul)
        d3.update(t, rp)
        d4.update(t, ud)
        d5.update(t, tde)
        #d1.update(t, [ang * 10, gvfs[5].get_prediction(state_prime_off),0, ang])
        #servo1_data.append([t, ang * 10, theta[state_prime], gamma(state_prime) * 3, cummlant(state_prime) *6])

        # simple pavlov
        #print gvfs[1].get_prediction(state_prime_on)
        if is_pavlov:
            print gvfs[0].get_prediction(state_prime_on)
            if gvfs[0].get_prediction(state_prime_on) > 7:
                pavlov(s2,ang)
        # go to the next time step
        t += 1
        last_state_off = current_state_off
        last_state_on = current_state_on
        last_time = time.time()
        # set last cumulant
        last_angle = ang
        last_load = load
        last_temp = temperature
        last_voltage = voltage
        last_time = time.time()
        if flag_stop:
            thread.exit_thread()


# write plotting data to file before ending by ctrl+c
def signal_handler(signal, frame):
    global flag_stop, servo1_data, spreds, scumul, srp, sud, stde

    # stop threads
    flag_stop = True

    # now we need to dump the sensorimotor datastream to disk
    npspreds = np.asarray(spreds)
    scumul = np.asarray(scumul)
    srp = np.asarray(srp)
    sud = np.asarray(sud)
    stde = np.asarray(stde)

    # with open('cumulants.txt', 'wb') as fp:
    #     pickle.dump(scumul, fp)
    #
    # with open('TDERROR.txt', 'wb') as fp:
    #     pickle.dump(stde, fp)

    np.savetxt('predictions.txt', npspreds)
    np.savetxt('cumulants.txt', scumul)
    np.savetxt('RUPEE.txt', srp)
    np.savetxt('UDE.txt', sud)
    np.savetxt('TDERROR.txt', stde)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()