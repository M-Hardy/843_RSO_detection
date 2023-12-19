"""
make SPARK dataset to be YOLOv8 compatible
11 object classes
split training set into 2 sets, 80% for training and 20% for validation
use validation set as test set
"""

import os
import csv
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import utils
import numpy as np

CLASS_MAP = {
    "smart_1": 0,
    "cheops": 1,
    "lisa_pathfinder": 2,
    "debris": 3,
    "proba_3_ocs": 4,
    "proba_3_csc": 5,
    "soho": 6,
    "earth_observation_sat_1": 7, 
    "proba_2": 8,
    "xmm_newton": 9,
    "double_star": 10
}

DATASET = "train"

def get_normalized_yolo_bbox_values(bbox_arr):
    #SPARK bbox cells follow the format: [R_min,C_min,R_max,C_max], where R refers to row, and C refers to column.
    width = length = 1024
    r_min, c_min, r_max, c_max = bbox_arr
    normalized_r_min, normalized_c_min, normalized_r_max, normalized_c_max = r_min/width, c_min/length, r_max/width, c_max/length
    normalized_yolo_bbox_values = utils.ops.xyxy2xywh(np.array([normalized_r_min, normalized_c_min, normalized_r_max, normalized_c_max]))
    return normalized_yolo_bbox_values

def move_image_labels(source_folder, destination_folder, validation_folder):
    validation_files = [os.path.splitext(file)[0] for file in os.listdir(validation_folder)]
    os.makedirs(destination_folder, exist_ok=True)
    for file_name in validation_files:
        source_txt_path = os.path.join(source_folder, file_name + ".txt") 
        if os.path.exists(source_txt_path): #check if file exists = check if path exists
            destination_txt_path = os.path.join(destination_folder, file_name + ".txt")
            shutil.move(source_txt_path, destination_txt_path)

def split_and_move_images(source_folder, destination_folder, validation_percent=20):
    image_files = [file for file in os.listdir(source_folder) if file.endswith(('.jpg', '.png', '.jpeg'))]
    train_files, validation_files = train_test_split(image_files, test_size=validation_percent / 100.0, random_state=42)

    os.makedirs(destination_folder, exist_ok=True)

    for file_name in validation_files:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.move(source_path, destination_path)

def write_txt_file(folder, filename, class_label, bbox, class_mapping):
    txt_filename = os.path.join(folder, filename + ".txt")
    with open(txt_filename, "w") as txt_file:
        txt_file.write(f"{class_mapping[class_label]} {' '.join(map(str, bbox))}")

def process_csv(destination_folder_name, csv_filename, class_mapping):
    with open(csv_filename, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            filename = row["filename"][:-4] #strip ".png" from filename value to get correct filename
            class_label = row["class"]
            bbox = row["bbox"].strip("[]").split(", ")
            bbox = [float(val) for val in bbox]  
            bbox = get_normalized_yolo_bbox_values(bbox)
            
            folder = os.path.join(os.path.dirname(csv_filename), destination_folder_name)
            os.makedirs(folder, exist_ok=True)
            write_txt_file(folder, filename, class_label, bbox, class_mapping)

if __name__ == "__main__":
    #csv_filename = f"../data/labels/{DATASET}.csv"  
    #process_csv(DATASET, csv_filename, CLASS_MAP)

    image_source_folder = "../data/images/train"  
    image_destination_folder = "../data/images/val"
    label_source_folder = "../data/labels/train"
    label_destination_folder = "../data/labels/val"  
    label_validation_folder = "../data/images/val"

    split_and_move_images(image_source_folder, image_destination_folder)
    move_image_labels(label_source_folder, label_destination_folder, label_validation_folder)