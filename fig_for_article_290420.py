# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 19:42:01 2020

@author: Amir
"""

# Needed functions
from ui_utils import curve_line


def matchPartners(subject1=None, filesToProcessS2=None, locationSettings=None):
    # Test if all vars were defined
    assert len([t for t in [subject1, filesToProcessS2, locationSettings] if
                t is None]) == 0, "one of the vars was not defined"

    # Loading subjects
    s1 = subject1
    # Getting the matching file - subject 2
    if isinstance(filesToProcessS2, list):
        s2 = filesToProcessS2[filesToProcessS2.index(s1.replace("SUBJECT1", "SUBJECT2"))]
    else:
        s2 = filesToProcessS2

    #    print(s1, "\n", s2)

    # Load files
    epochsS1 = mne.read_epochs(s1, preload=True, verbose=False)
    epochsS2 = mne.read_epochs(s2, preload=True, verbose=False)

    assert epochsS1.info["ch_names"] == epochsS2.info["ch_names"], "files don't have the same channels order"

    # Rename electrodes names
    epochsS1.rename_channels(dict(zip(epochsS1.info["ch_names"], [i + "-0" for i in epochsS1.info["ch_names"]])))
    epochsS2.rename_channels(dict(zip(epochsS2.info["ch_names"], [i + "-1" for i in epochsS2.info["ch_names"]])))

    # Combining the subjects to one cap
    combined = combineEpochs(epochsS1=epochsS1, epochsS2=epochsS2)

    # Adding sensors(channles) locations
    if type(combined) is not str:
        combined.info["chs"] = locationSettings.copy()

    return combined


def matchDescription(epochs=None, description=None):
    assert len(epochs.drop_log) == len(description), "drop_log and description not the same length"

    indexDrop = []
    goodCounter = -1
    for i in range(0, len(epochs.drop_log)):
        # Testing for good data in subject
        if any(epochs.drop_log[i]) == False:
            # Counter of good data, for indexing the epochs file
            goodCounter += 1
            # Testing to see if in the description the data is bad
            if description[i] == ["bad data"]:
                # Adding the index goodCounter to the list for droping
                indexDrop.append(goodCounter)

    # Droping the bad epochs
    epochs.drop(indexDrop, reason='bad combined')

    epochs.drop_bad()
    return epochs


def combineEpochs(epochsS1=None, epochsS2=None):
    assert len(epochsS1.drop_log) == len(epochsS2.drop_log), "drop_log not the same length"

    # Creating the list of the good/bad epochs in both subjects
    description = []
    for logA, logB in zip(epochsS1.drop_log, epochsS2.drop_log):
        if (any(logA) == True) | (any(logB) == True):
            description.append(["bad data"])
        else:
            description.append(["good data"])

    if len([d for d in description if d == ["good data"]]) < 5:
        return ("Not enoguh good data")

    # Matching that bad/good epochs for each particiant. It's the intersection of good epochs
    matchDescription(epochs=epochsS1, description=description)
    matchDescription(epochs=epochsS2, description=description)

    # Combine matched epochs as one cap, for that I rebuild the two epochs as one epoch structure from scratch
    ##concatenating the data from the caps as one
    # Test if epochs are in the same freq. If not, downsample the higher to 500fq
    if epochsS1.info["sfreq"] != epochsS2.info["sfreq"]:
        # set Sample rate
        if min(epochsS1.info["sfreq"], epochsS2.info["sfreq"]) == 500:
            newsampleRate = 250
        else:
            for i in range(int(min(epochsS1.info["sfreq"], epochsS2.info["sfreq"])), 100, -1):
                if epochsS1.copy().resample(i, npad='auto').to_data_frame().shape == epochsS2.copy().resample(i,
                                                                                                              npad='auto').to_data_frame().shape:
                    newsampleRate = i
                    break

        epochsS1.resample(newsampleRate, npad='auto')
        epochsS2.resample(newsampleRate, npad='auto')

        # Concatenate epochs from subject 1 and subject2
    data = np.concatenate((epochsS1, epochsS2), axis=1)
    ##Creating an info structure
    info = mne.create_info(
        ch_names=list(epochsS1.info["ch_names"] + epochsS2.info["ch_names"]),
        ch_types=np.repeat("eeg", len(list(epochsS1.info["ch_names"] + epochsS2.info["ch_names"]))),
        sfreq=epochsS1.info["sfreq"])
    ##Creating an events structure
    events = np.zeros((data.shape[0], 3), dtype="int32")

    # Naming the events by the name of of the original epoch number.
    # e.g. event == 289 is epoch 289 in the original data
    eventConter = 0
    for i, d in enumerate(description):
        if d == ['good data']:
            events[eventConter][0] = i
            events[eventConter][2] = i
            eventConter += 1
    ##Creating event ID
    event_id = dict(zip([str(item[2]) for item in events], [item[2] for item in events]))
    ##Time of each epoch
    tmin = -0.5
    ##Building the epoch structure
    combined = mne.EpochsArray(data, info, events, tmin, event_id)
    ##Editing the channels locations
    combined.info["chs"] = epochsS1.info["chs"] + epochsS2.info["chs"]

    # test to see that all the good epochs are in the same length
    if len(set(map(len, [epochsS1, epochsS2, combined]))) == 1 and sum(
            [i == ['good data'] for i in description]) == len(epochsS1):
        print("All are the same length")
    else:
        print("ERROR - They are not the same length!")

    return combined


# Loading modules and setting up paths
import os, sys
import mne
import pandas as pd
import pickle
import numpy as np
from scipy import linalg, stats
from mne import io
from mne.connectivity import spectral_connectivity
import winsound
import warnings
import re
from mne.channels import find_ch_adjacency  # Used to be find_ch_connectivity
import matplotlib.pyplot as plt
# from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix

# I know it's no the correct way of doing things.
path = "data"
fileExt = ".fif"

# Getting all the files in the folder
listOfFiles = [x[0] + "/" + f for x in os.walk(path) for f in x[2] if
               f.endswith(fileExt)]

# This is the prerequisites.
# Here I'm uploading files two participants, combining them to one, and calculated their
# new electrodes physical positions for later viz.

# locations for cap1 and cap2 in the combined file
from mpl_toolkits.mplot3d import Axes3D
from copy import copy

# Creating combined object
s1 = listOfFiles[0]
s2 = listOfFiles[listOfFiles.index(s1.replace("SUBJECT1", "SUBJECT2"))]
epochsS1 = mne.read_epochs(s1, preload=True)
epochsS2 = mne.read_epochs(s2, preload=True)
epochsS1.rename_channels(dict(zip(epochsS1.info["ch_names"], [i + "-0" for i in epochsS1.info["ch_names"]])))
epochsS2.rename_channels(dict(zip(epochsS2.info["ch_names"], [i + "-1" for i in epochsS2.info["ch_names"]])))
combined = combineEpochs(epochsS1=epochsS1, epochsS2=epochsS2)

# Calculating locations
locations = copy(np.array([ch['loc'] for ch in combined.info['chs']]))
cap1_locations = locations[:31, :3]
print("Mean: ", np.nanmean(cap1_locations, axis=0))
print("Min: ", np.nanmin(cap1_locations, axis=0))
print("Max: ", np.nanmax(cap1_locations, axis=0))

translate = [0, 0.25, 0]
rotZ = np.pi

cap2_locations = copy(cap1_locations)
newX = cap2_locations[:, 0] * np.cos(rotZ) - cap2_locations[:, 1] * np.sin(rotZ)
newY = cap2_locations[:, 0] * np.sin(rotZ) + cap2_locations[:, 1] * np.cos(rotZ)
cap2_locations[:, 0] = newX
cap2_locations[:, 1] = newY
cap2_locations = cap2_locations + translate
print("Mean: ", np.nanmean(cap2_locations, axis=0))
print("Min: ", np.nanmin(cap2_locations, axis=0))
print("Max: ", np.nanmax(cap2_locations, axis=0))
sens_loc = np.concatenate((cap1_locations, cap2_locations), axis=0)

# testing that the new locations and the old locations are at the same length
assert len([ch['loc'] for ch in combined.info['chs']]) == len(sens_loc), "the caps locations are not in the same length"

# Changing location
for old, new in enumerate(sens_loc):
    combined.info["chs"][old]["loc"][0:3] = new[0:3]

locationSettings = combined.info["chs"].copy()

del cap1_locations, cap2_locations, old, new, newX, newY, rotZ, translate, s1, s2


##Plot two caps connectivity
def capTest(x):
    if x >= 31:
        out = x - 31
    else:
        out = x
    return (out)


# Brain areas
centerS1 = ['Cz-0', 'C3-0', 'C4-0']
leftTemporalS1 = ["FT9-0", "TP9-0", "T7-0"]
rightTemporalS1 = ["FT10-0", "TP10-0", "T8-0"]

chToTake = [i[:-2] for i in centerS1 + leftTemporalS1 + rightTemporalS1]

# areas = [[i[:-2] + "-1" for i in centerS1] + centerS1,
#         [i[:-2] + "-1" for i in leftTemporalS1] + leftTemporalS1, 
#         [i[:-2] + "-1" for i in rightTemporalS1] + rightTemporalS1]

areas = [[i[:-2] for i in centerS1],
         [i[:-2] for i in leftTemporalS1],
         [i[:-2] for i in rightTemporalS1]]

sensloc = np.array([c['loc'][:3] for c in combined.info['chs']][:62])

#####################Creating plot for example####################
import mayavi.mlab as mlab

# import moviepy.editor as mpy

con = np.zeros([62, 62])
con.size
Itook = list()
for e1 in range(62):
    for e2 in range(62):
        for area in areas:
            i = 1
            # Between
            #            if combined.info["ch_names"][e1][:-2] in area and combined.info["ch_names"][e2][:-2] in area:
            #                k1, k2 = list(map(capTest, [e1, e2]))
            #                if e1 <= 30 and e2 >= 31:
            #                    con[e1][e2] = 0.5
            #                    Itook.append([combined.info["ch_names"][e1], e1, combined.info["ch_names"][e2], e2])

            # Cap1
            if combined.info["ch_names"][e1][:-2] in area and combined.info["ch_names"][e2][:-2] in area:
                k1, k2 = list(map(capTest, [e1, e2]))
                if e1 <= 30 and e2 <= 30 and e1 != e2:
                    con[e1][e2] = 1
            #                    Itook.append([combined.info["ch_names"][e1], e1, combined.info["ch_names"][e2], e2])
            # Cap2
            if combined.info["ch_names"][e1][:-2] in area and combined.info["ch_names"][e2][:-2] in area:
                k1, k2 = list(map(capTest, [e1, e2]))
                if e1 >= 31 and e2 >= 31 and e1 != e2:
                    con[e1][e2] = 1
    #                    Itook.append([combined.info["ch_names"][e1], e1, combined.info["ch_names"][e2], e2])

    sum(con == 1)

A = csr_matrix(con.tolist())

mlab.clf()
# Ploting caps
fig = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
points = mlab.points3d(sens_loc[:, 0], sens_loc[:, 1], sens_loc[:, 2],
                       color=(0.5, 0.5, 0.5), opacity=1, scale_factor=0.005,
                       figure=fig)

# Set view
mlab.view(azimuth=180, distance=0.7, focalpoint="auto")

#######
# Get the strongest connections
n_con = len(con) ** 2  # show up to 3844 connections
min_dist = 0  # exclude sensors that are less than 5cm apart
threshold = np.sort(con, axis=None)[-n_con]  # sort the con by size and pick the index of n_con
ii, jj = np.where(con > 0)

# Remove close connections
con_nodes = list()
con_val = list()
for i, j in zip(ii, jj):
    if linalg.norm(sens_loc[i] - sens_loc[j]) > min_dist:
        con_nodes.append((i, j))
        con_val.append(con[i, j])

con_val = np.array(con_val)

# Show the connections as tubes between sensors

# By General - all in the same color.

# TODO Here I'm creating triangles for the areas I averaged.
# However, I need to reverse this to from one electrode to another.
vmax = np.max(con_val)
vmin = np.min(con_val)
for val, nodes in zip(con_val, con_nodes):
    x1, y1, z1 = sens_loc[nodes[0]]
    x2, y2, z2 = sens_loc[nodes[1]]
    lines = mlab.plot3d([x1, x2], [y1, y2], [z1, z2], [val, val],
                        vmin=vmin, vmax=vmax, tube_radius=0.0002,
                        colormap='blue-red')
    lines.module_manager.scalar_lut_manager.reverse_lut = True

# Creating tיe lines from the center of each triangle.
for area, color in zip(areas, [(1, 0, 0),  # central
                               (0, 1, 0),  # left
                               (0, 0, 1)]):  # right
    # subject1
    for a in area:
        x1, y1, z1 = np.array([sens_loc[combined.info["ch_names"].index(a + "-0")] for a in area]).mean(axis=0)
    # subject1
    for a in area:
        x2, y2, z2 = np.array([sens_loc[combined.info["ch_names"].index(a + "-1")] for a in area]).mean(axis=0)

        x, y, z = curve_line(p1=np.array([x1, y1, z1]), p2=np.array([x2, y2, z2]), amp=0.2)
        linesTriangle = mlab.plot3d(x, y, z, vmin=vmin, vmax=vmax, tube_radius=0.002, color=color)

    ## Add the sensor names for the connections shown
# nodes_shown = list(set([n[0] for n in con_nodes] +
#                       [n[1] for n in con_nodes]))

nodes_shown = list(range(0, 62))

chNames = []
# Changing channels name as letters -M / -F
for i, c in enumerate(combined.info["ch_names"]):
    if c[:-2] in chToTake:
        if c[-1] == "0":
            chNames.append(c[:-1] + "M")
        elif c[-1] == "1":
            chNames.append(c[:-1] + "F")

# Channels name as letters -M / -F
picks = np.array(list(range(0, len(chNames))))

counterif = -1
for i, node in enumerate(nodes_shown):
    if combined.info["ch_names"][i][:-2] in chToTake:
        counterif += 1
        x, y, z = sens_loc[i]
        mlab.text3d(x, y, z, chNames[counterif],
                    scale=0.005,
                    color=(0, 0, 0))

mlab.show()
# TODO
# HERE I need to add the brains under each cap
# MNE's brains are actually taken from here: https://pysurfer.github.io
# Brain example can be found here: https://mne.tools/stable/auto_tutorials/preprocessing/plot_70_fnirs_processing.html#view-location-of-sensors-over-brain-surface
