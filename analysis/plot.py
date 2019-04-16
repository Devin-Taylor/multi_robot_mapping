import json
import os
from glob import glob

from matplotlib import pyplot as plt
import numpy as np

ROOT = "/home/devin/catkin_ws/src/multi_robot"
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(RESULTS, "plots")

if not os.path.exists(PLOTS):
    os.mkdir(PLOTS)

def get_acc_mean_std(arr):
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

def robot_group_performance(num_robots, decentralised, origin_known, frame_cap=200):

    files = glob(os.path.join(RESULTS, "results_numbots-{}_decentralised-{}_originknown-{}_run-*".format(num_robots, decentralised, origin_known)))

    data = []
    for f in files:
        with open(f) as fd:
            data.append(json.load(fd))

    all_frames = [max(x['frame']) for x in data]
    max_frame = min(np.sort(all_frames)[1], frame_cap)
    frames = list(range(1, max_frame+1))

    def get_acc_array(arr, key):
        accuracy = np.empty((len(files), max_frame))
        accuracy[:] = np.nan
        for idx in range(len(arr)):
            accuracy[idx, :min(all_frames[idx], max_frame)] = arr[idx][key][:min(all_frames[idx], max_frame)]
        return accuracy


    accuracy_mu, accuracy_std = get_acc_mean_std(get_acc_array(data, 'accuracy')) # np.array([x['accuracy'][:max_frame] for x in data]))
    if len(data[0]['r2_accuracy']):
        bot1_mu, bot1_std = get_acc_mean_std(get_acc_array(data, 'r1_accuracy')) # np.array([x['r1_accuracy'][:max_frame] for x in data]))
        bot2_mu, bot2_std = get_acc_mean_std(get_acc_array(data, 'r2_accuracy')) # np.array([x['r2_accuracy'][:max_frame] for x in data]))
    if len(data[0]['r3_accuracy']):
        bot3_mu, bot3_std = get_acc_mean_std(get_acc_array(data, 'r3_accuracy')) # np.array([x['r3_accuracy'][:max_frame] for x in data]))


    plt.fill_between(frames, accuracy_mu-2*accuracy_std, accuracy_mu+2*accuracy_std, alpha=.1)
    plt.plot(frames, accuracy_mu)
    plt.xlabel("Frame number")
    plt.ylabel("Dice coefficient")
    if len(data[0]['r2_accuracy']):
        plt.fill_between(frames, bot1_mu-2*bot1_std, bot1_mu+2*bot1_std, alpha=.1)
        plt.plot(frames, bot1_mu)
        plt.fill_between(frames, bot2_mu-2*bot2_std, bot2_mu+2*bot2_std, alpha=.1)
        plt.plot(frames, bot2_mu)
        plt.legend(["Overall", "Robot 1", "Robot 2"])
    if len(data[0]['r3_accuracy']):
        plt.fill_between(frames, bot3_mu-2*bot3_std, bot3_mu+2*bot3_std, alpha=.1)
        plt.plot(frames, bot3_mu)
        plt.legend(["Overall", "Robot 1", "Robot 2", "Robot 3"])
    plt.savefig(os.path.join(PLOTS, "group_performance_numbots-{}_decentralised-{}_originknown-{}.png".format(num_robots, decentralised, origin_known)), bbox_inches='tight', format="png", dpi=300)
    plt.show()

def robot_number_comparison(decentralised, origin_known, frame_cap=200):

    robot_files = []
    files = glob(os.path.join(RESULTS, "results_numbots-{}_decentralised-{}_originknown-{}_run-*".format(1, False, True)))
    robot_files.append(files)
    for r in range(2, 4):
        files = glob(os.path.join(RESULTS, "results_numbots-{}_decentralised-{}_originknown-{}_run-*".format(r, decentralised, origin_known)))
        robot_files.append(files)

    robot_mus = []
    robot_stds = []
    for files in robot_files:
        data = []
        for f in files:
            with open(f) as fd:
                data.append(json.load(fd))

        frames = [max(x['frame']) for x in data]
        max_frame = min(np.sort(frames)[1], frame_cap)
        accuracy = np.empty((len(files), max_frame))
        accuracy[:] = np.nan
        for idx in range(len(data)):
            accuracy[idx, :min(frames[idx], max_frame)] = data[idx]['accuracy'][:min(frames[idx], max_frame)]
        frames = list(range(1, max_frame+1))

        accuracy_mu, accuracy_std = get_acc_mean_std(accuracy) # np.array([x['accuracy'][:max_frame] for x in data]
        plt.plot(frames, accuracy_mu)
        plt.fill_between(frames, accuracy_mu-2*accuracy_std, accuracy_mu+2*accuracy_std, alpha=.1)


    plt.xlabel("Frame number")
    plt.ylabel("Dice coefficient")
    plt.legend(["1 Robot", "2 Robots", "3 Robots"])
    plt.savefig(os.path.join(PLOTS, "number_comparison_decentralised-{}_originknown-{}.png".format(decentralised, origin_known)), bbox_inches='tight', format="png", dpi=300)
    plt.show()

def robot_scenario_comparison(num_robots, frame_cap=200):

    robot_files = []
    robot_files.append(glob(os.path.join(RESULTS, "results_numbots-{}_decentralised-{}_originknown-{}_run-*".format(num_robots, False, True))))
    robot_files.append(glob(os.path.join(RESULTS, "results_numbots-{}_decentralised-{}_originknown-{}_run-*".format(num_robots, False, False))))
    robot_files.append(glob(os.path.join(RESULTS, "results_numbots-{}_decentralised-{}_originknown-{}_run-*".format(num_robots, True, True))))
    robot_files.append(glob(os.path.join(RESULTS, "results_numbots-{}_decentralised-{}_originknown-{}_run-*".format(num_robots, True, False))))

    robot_mus = []
    robot_stds = []
    for files in robot_files:
        data = []
        for f in files:
            with open(f) as fd:
                data.append(json.load(fd))

        all_frames = [max(x['frame']) for x in data]
        max_frame = min(np.sort(all_frames)[1], frame_cap)
        accuracy = np.empty((len(files), max_frame))
        accuracy[:] = np.nan
        for idx in range(len(data)):
            accuracy[idx, :min(all_frames[idx], max_frame)] = data[idx]['accuracy'][:min(all_frames[idx], max_frame)]
        frames = list(range(1, max_frame+1))

        accuracy_mu, accuracy_std = get_acc_mean_std(accuracy) # np.array([x['accuracy'][:max_frame] for x in data]))
        plt.plot(frames, accuracy_mu)
        plt.fill_between(frames, accuracy_mu-2*accuracy_std, accuracy_mu+2*accuracy_std, alpha=.1)


    plt.xlabel("Frame number")
    plt.ylabel("Dice coefficient")
    plt.legend(["Centralised, origin known", "Centralised, origin unknown", "Decentralised, origin known", "Decentralised, origin unknown"])
    plt.savefig(os.path.join(PLOTS, "scenario_comparison_numbots-{}.png".format(num_robots)), bbox_inches='tight', format="png", dpi=300)
    plt.show()

def accuracy_evaluation(decentralised, origin_known, frame_cap=1000, max_args=False):

    robot_files = []
    files = glob(os.path.join(RESULTS, "results_numbots-{}_decentralised-{}_originknown-{}_run-*".format(1, False, True)))
    robot_files.append(files)
    for r in range(2, 4):
        files = glob(os.path.join(RESULTS, "results_numbots-{}_decentralised-{}_originknown-{}_run-*".format(r, decentralised, origin_known)))
        robot_files.append(files)

    robot_mus = []
    robot_stds = []
    robot_acc = []
    for files in robot_files:
        data = []
        for f in files:
            with open(f) as fd:
                data.append(json.load(fd))

        frames = [max(x['frame']) for x in data]
        max_frame = min(np.sort(frames)[1], frame_cap)
        accuracy = np.empty((len(files), max_frame))
        accuracy[:] = np.nan
        for idx in range(len(data)):
            accuracy[idx, :min(frames[idx], max_frame)] = data[idx]['accuracy'][:min(frames[idx], max_frame)]
        frames = list(range(1, max_frame+1))

        accuracy_mu, accuracy_std = get_acc_mean_std(accuracy) # np.array([x['accuracy'][:max_frame] for x in data]
        if max_args:
            print(np.argmax(accuracy_mu))
        else:
            print(max(accuracy_mu))
