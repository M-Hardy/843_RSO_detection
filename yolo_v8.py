from ultralytics import YOLO
import torch
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

#DATASET = "train"
DATASET_PATH = "SPARK/images/all"

def create_val_set(source_folder):
    # Create the source folder if it doesn't exist
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)

    destination_folder = 'SPARK/images/val'
    # Create destination folders if they don't exist
    for folder in destination_folder:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Get a list of all files in the source folder
    files = os.listdir(source_folder)

    # Calculate the number of files to move for each destination folder
    total_files = len(files)

    for i in range(0, total_files, 4):
        file_to_move = files[i]    
        source_path = os.path.join(source_folder, file_to_move)
        destination_path = os.path.join(folder, file_to_move)
        shutil.move(source_path, destination_path)


def halve_dataset(source_folder):
    # Create the source folder if it doesn't exist
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)
    
    destination_folder = source_folder + "-2nd_half"

    # Create the destination folder if it doesn't exist
    #if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

    # Get a list of all files in the source folder
    files = os.listdir(source_folder)

    # Iterate over every other file and move it to the destination folder
    for i in range(1, len(files), 2):
        file_to_move = files[i]
        source_path = os.path.join(source_folder, file_to_move)
        destination_path = os.path.join(destination_folder, file_to_move)

        # Move the file
        shutil.move(source_path, destination_path)


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

def split_labels(image_train, image_label, all_labels):
    train_label_folder = "SPARK/labels/train"
    val_label_folder = "SPARK/labels/val"  
    remainder_label_folder = "SPARK/labels/images_2nd_half"  
    
    labels = os.listdir(all_labels)
    labels_stripped = [label[:-4] for label in labels]
    img_train = [filename[:-4] for filename in os.listdir(image_train)]
    img_val = [filename[:-4] for filename in os.listdir(image_label)]
    
    for i in range(len(labels)):
        label = labels_stripped[i]
        label_file = labels[i]
                
        source_path = os.path.join(all_labels, label_file)

        if label in img_train:
            destination_path = os.path.join(train_label_folder, label_file)
        elif label in img_val:
            destination_path = os.path.join(val_label_folder, label_file)
        else:
            destination_path = os.path.join(remainder_label_folder, label_file)

        # Move the file
        shutil.move(source_path, destination_path)


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

def run_yolov8_training():
    torch.cuda.set_device(0) # Set to your desired GPU number
    model = YOLO('yolov8n.pt')
    results = model.train(data='config.yaml', epochs=5, device=0, imgsz=160, batch=-1)

if __name__=='__main__':
    #csv_filename = f"../data/labels/{DATASET}.csv"  
    #process_csv(DATASET, csv_filename, CLASS_MAP)

    image_train = "SPARK/images/train"  
    image_val = "SPARK/images/val"
    label_source_folder = "SPARK/labels/all"
    #label_destination_folder = "SPARK/labels/val"  
    #label_validation_folder = "SPARK/images/val"

    # split_and_move_images(image_source_folder, image_destination_folder)
    #move_image_labels(label_source_folder, label_destination_folder, label_validation_folder)
    #split_labels(image_train, image_val, label_source_folder)

    #halve_dataset(DATASET_PATH)
    #create_val_set(DATASET_PATH)
    run_yolov8_training()