#!/usr/bin/env python
# coding: utf-8

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import util.steering as steering
import util.transformation as transformation
from util.JSONReader import *

import pandas as pd
import pickle as pkl
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import PIL.Image as pil
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import argparse
import random

# set seed
np.random.seed(0)
random.seed(0)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, help="path to the directory containing the raw dataset (.mov & .json)")
parser.add_argument("--train", type=str, help="path to the text file containing the training scenes")
args = parser.parse_args()


def get_steer(course, speed, dt, eps=1e-12):
    sgn = np.sign(course)
    dist = speed * dt
    R = dist / (np.deg2rad(abs(course)) + eps)
    delta, _, _ = steering.get_delta_from_radius(R)
    steer = sgn * steering.get_steer_from_delta(delta)
    return steer


def get_course(steer, speed, dt):
    dist = speed * dt
    delta = steering.get_delta_from_steer(steer)
    R = steering.get_radius_from_delta(delta)
    rad_course = dist / R
    course = np.rad2deg(rad_course)
    return course


def augment(data, translation, rotation, intersection_distance=7.5):
    """
    Augment a frame
    Warning!!! this augmentation may work only for turns less than 180 degrees. For bigger turns, although it
    reaches the same point, it may not follow the real car's trajectory

    :param data: [steer, velocity, delta_time]
    :param translation: ox translation, be aware that positive values mean right translation
    :param rotation: rotation angle, be aware that positive valuea mean right rotation
    :param intersection_distance: distance where the simualted car and real car will intersect
    :return: the augmented frame, steer for augmented frame
    """
    assert abs(rotation) < math.pi / 2, "The angle in absolute value must be less than Pi/2"

    steer, _, _ = data
    eps = 1e-12

    # compute wheel angle and radius of the real car
    steer = eps if abs(steer) < eps else steer
    wheel_angle = steering.get_delta_from_steer(steer + eps)
    R = steering.get_radius_from_delta(wheel_angle)

    # estimate the future position of the real car
    alpha = intersection_distance / R  # or may try velocity * delta_time / R
    P1 = np.array([R * (1 - np.cos(alpha)), R * np.sin(alpha)])

    # determine the point where the simulated car is
    P2 = np.array([translation, 0.0])

    # compute the line parameters that passes through simulated point and is
    # perpendicular to it's orientation
    d = np.zeros((3,))
    rotation = eps if abs(rotation) < eps else rotation
    d[0] = np.sin(rotation)
    d[1] = np.cos(rotation)
    d[2] = -d[0] * translation

    # we need to find the circle center (Cx, Cy) for the simulated car
    # we have the equations
    # (P11 - Cx)**2 + (P12 - Cy)**2 = (P21 - Cx)**2 + (P22 - Cy)**2
    # d0 * Cx + d1 * Cy + d2 = 0
    # to solve this, we substitute Cy with -d0/d1 * Cx - d2/d1
    a = P1[0]**2 + (P1[1] + d[2]/d[1])**2 - P2[0]**2 - (P2[1] + d[2]/d[1])**2
    b = -2 * P2[0] + 2 * d[0]/d[1] * (P2[1] + d[2]/d[1]) + 2 * P1[0] - 2 * d[0]/d[1] * (P1[1] + d[2]/d[1])
    Cx = a / b
    Cy = -d[0]/d[1] * Cx - d[2]/d[1]
    C = np.array([Cx, Cy])

    # determine the radius
    sim_R = np.linalg.norm(C - P2)
    assert np.isclose(sim_R, np.linalg.norm(C - P1)), "The points P1 and P2 are not on the same circle"

    # determine the "sign" of the radius
    # sgn = 1 if np.cross(w2, w1) >= 0 else -1
    w1 = np.array([np.sin(rotation), np.cos(rotation)])
    w2 = P1 - P2
    sgn = 1 if np.cross(w2, w1) >= 0 else -1
    sim_R = sgn * sim_R

    # determine wheel angle
    sim_delta, _, _ = steering.get_delta_from_radius(sim_R)
    sim_steer = steering.get_steer_from_delta(sim_delta)
    return sim_steer, sim_delta, sim_R, C


def pipeline(img: np.array, tx: float=0.0, ry: float=0.0):
    # convension
    tx, ry = -tx, -ry
    
    # transform image to tensor
    img = np.asarray(img)
    height, width = img.shape[:2]
    
    K = np.array([
        [0.61, 0, 0.5],   # width
        [0, 1.09, 0.5],   # height
        [0, 0, 1]])
    K[0, :] *= width 
    K[1, :] *= height
    
    M = np.array([
        [1,  0, 0, 0.00],
        [0, -1, 0, 1.65],
        [0,  0, 1, 1.54],
        [0, 0, 0, 1]
    ])
    M = np.linalg.inv(M)[:3, :]
    
    # transformation object
    transform = transformation.Transformation(K, M)
    
    output = transform.rotate_image(img, ry)
    output = transform.translate_image(output, tx)
    output = output[:320, ...]
    output = cv2.resize(output, (256, 128))
    
    # crop
    crop = transformation.Crop()
    output = crop.crop_center(output, up=0.1, down=0.5, left=0.25, right=0.25)
    return output


def read_json(root_dir: str, json: str, verbose: bool = False):
    json_reader = JSONReader(root_dir, json, frame_rate=3)
    crop = transformation.Crop()
    frame_idx = 0
    
    while True:
        # get next frame corresponding to current prediction
        frame, speed, rel_course = json_reader.get_next_image()
        if frame.size == 0:
            break
        
        dt = 0.333

        steer = get_steer(rel_course, speed, dt=dt)
        tx, ry = 0.0, 0.0
        sgnt = 1 if np.random.rand() > 0.5 else -1
        sgnr = 1 if np.random.rand() > 0.5 else -1

        # generate random transformation
        if np.random.rand() < 0.33:
            tx = sgnt * np.random.uniform(0.25, 0.75)
            ry = sgnr * np.random.uniform(0.05, 0.1)
        else:
            if np.random.rand() < 0.5:
                tx = sgnt * np.random.uniform(0.25, 0.75)
            else:
                ry = sgnr * np.random.uniform(0.05, 0.1) 
        
        aug_name = "%s.tx=%.2f.ry=%.2f" % (json[:-5], tx, ry) 
    
        # generate augmented image
        aug_img = pipeline(img=frame, tx=tx, ry=ry)

        # generate augmented steering comand
        aug_steer, _, _, _ = augment(
            data=[steer, speed, dt],
            translation=tx,
            rotation=ry,
        )

        # convert steer to course
        aug_course = get_course(aug_steer, speed, dt)
        
        # save image and data
        scene = json[:-5]
        
        frame_path = os.path.join("../dataset/img_aug", scene + "." + str(frame_idx) + ".png")
        cv2.imwrite(frame_path, aug_img)
       
        data_path = os.path.join("../dataset/data_aug", scene + "." + str(frame_idx) + ".pkl")
        with open(data_path, "wb") as fout:
            pkl.dump({"speed": speed, "rel_course": aug_course, "tx": tx, "ry": ry}, fout)
        
        frame_idx += 1
        
        if verbose == True:
            print("Speed: %.2f, Relative Course: %.2f" % (speed, rel_course))
            print("Course: %.2f", aug_course)
            print("Frame shape:", aug_img.shape)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(aug_img[..., ::-1])
            ax2.imshow(img_map[..., ::-1])
            plt.show()


if __name__ == "__main__":
	# get train scenes
	with open(args.train, "rt") as fin:
	    train_scenes = fin.read()
	train_scenes = set(train_scenes.split("\n"))

	files = os.listdir(args.root_dir)
	jsons = [file for file in files if file.endswith(".json") and file[:-5] in train_scenes]

	# creat necessary direcotries
	if not os.path.exists("../dataset"):
		os.makedirs("../dataset")

	if not os.path.exists("../dataset/img_aug"):
		os.makedirs("../dataset/img_aug")

	if not os.path.exists("../dataset/data_aug"):
		os.makedirs("../dataset/data_aug")

	for json in tqdm(jsons):
	    read_json(args.root_dir, json, False)

	aug_files = os.listdir("../dataset/img_aug")
	aug_files = [file[:-4] for file in aug_files]

	df = pd.DataFrame(aug_files, columns=["name"])
	df.to_csv("../dataset/train_aug.csv", index=False)

	df = pd.read_csv("../dataset/train_aug.csv")
	df.head()




