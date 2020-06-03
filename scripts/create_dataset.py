#!/usr/bin/env python
# coding: utf-8
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from util.JSONReader import *
import util.transformation as transformation
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
from copy import deepcopy


def read_json(root_dir: str, json: str, verbose: bool = False):
	json_reader = JSONReader(root_dir, json, frame_rate=3)
	crop = transformation.Crop()
	predicted_course = 0.0
	frame_idx = 0
	
	while True:
		# get next frame corresponding to current prediction
		frame, speed, rel_course = json_reader.get_next_image()
		if frame.size == 0:
			break
			
		# process frame
		orig_frame = frame.copy()
		frame = frame[:320, ...]
		frame = cv2.resize(frame, (256, 128))
		frame = crop.crop_center(frame, up=0.1, down=0.5, left=0.25, right=0.25)

		# save image and data
		scene = json[:-5]
		
		frame_path = os.path.join("../dataset/img_real", scene + "." + str(frame_idx) + ".png")
		cv2.imwrite(frame_path, frame)
	   
		data_path = os.path.join("../dataset/data_real", scene + "." + str(frame_idx) + ".pkl")
		with open(data_path, "wb") as fout:
			pkl.dump({"speed": speed, "rel_course": rel_course}, fout)
		
		frame_idx += 1
		
		if verbose == True:
			print("Speed: %.2f, Relative Course: %.2f" % (speed, rel_course))
			print("Frame shape:", frame.shape)
			plt.imshow(frame[..., ::-1])
			plt.show()



if __name__ == "__main__":
	ROOT_DIR = "/mnt/storage/workspace/roberts/upb/all_3fps"
	files = os.listdir(ROOT_DIR)
	jsons = [file for file in files if file.endswith(".json")]


	# create necessary direcotrys
	if not os.path.exists("../dataset"):
		os.makedirs("../dataset")

	if not os.path.exists("../dataset/img_real"):
		os.makedirs("../dataset/img_real")

	if not os.path.exists("../dataset/data_real"):
		os.makedirs("../dataset/data_real")

	for json in tqdm(jsons):
		read_json(ROOT_DIR, json, False)

