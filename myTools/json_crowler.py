'''
version: 15.10.2024
'''



import numpy as np
#import pandas as pd
import os
import sys
import datetime
import json
import fnmatch
#import json_crowler
import random
from shutil import copy
import shutil
import cv2

label_studio_file_id_len = 9 # label studio creates and attaches an id before each upaloaded file name

##########################
# General helper functions
########
def create_file_transfer_list(directory, num_id_letters):
    '''
    creates a unique list of file ids, based on first "num_id_letters" of file names in the directory.
    This function assumes that filenames of files in the directory strat with IDs.
    '''
    filenames = [filename for filename in os.listdir(directory)]
    file_ids = [item[:num_id_letters] for item in filenames]
    unique_file_ids = [x for i, x in enumerate(file_ids) if x not in file_ids[:i]]
    return unique_file_ids

def file_organizer(source_directory, destination_directory, file_id_list, verbos = True, remove_exif = False):
    '''
    This function transfers files from source dir to the dest dir based on a list of ids as their starting charachters on the filename.
    If files are images, you can set "remove_exif" as True.
    '''
    # Get all file_ids from destination directory by checking json files in the dir
    source_filenames = [filename for filename in os.listdir(source_directory)]
    dest_filenames = [filename for filename in os.listdir(destination_directory)]
    for file_id in file_id_list:
        shortlist = [i for i in source_filenames if (i.startswith(file_id) and i not in dest_filenames)]
        for filename in shortlist:
            source_filepath = os.path.join(source_directory, filename)
            dest_filepath = os.path.join(destination_directory, filename)
            if verbos: print(filename)
            if remove_exif: #in case it is an image
                img = cv2.imread(source_filepath)  # OpenCV ignores EXIF metadata while reading.
                cv2.imwrite(dest_filepath, img)
            else:
                try:
                    copy(source_filepath, dest_filepath)
                except shutil.SameFileError as e:
                    print(e)
                    pass
    return True

#_______________________________
# End of general helper functions 
##################


def json_record_tooth_polygon_fetch(json_image_record, redundant_id_len, test_set = False, verbos = True):
    """fetch tooth polygon label data from image record in the json file.
      this function should be called from another loop that iterates all image records in the json file.

      json_image_record: record fetched from the json file. based on the known structure, each record has all the data for one image.
      redundant_id_len: label studio creates random id for each uploaded file. we need to remove this id from file name to make it compatible with local file names.
      """
    image_result_dict = {"image_filename":"", "image_perspective": "", "image_quality":"", "image_width": 0, "image_height": 0, "teeth_data": {} }

    output_data_valid = False # if the file contain valid data, this flag will rise.
    flag_image_data_recorded = False # to prevent overwriting image size data

    #get file name. some json files in the dataset have slightly different format. So, I use try except to handle them. if failed raises the except to the caller func.
    try:
         image_result_dict["image_filename"] = json_image_record['file_upload'][redundant_id_len:]
    except:
         try:
              image_result_dict["image_filename"] = get_substring_after_p5c(json_image_record['data']['image'])
         except:
              raise
     #find out the camera perspective
    if "front-" in image_result_dict["image_filename"]:
         image_result_dict["image_perspective"] = "front"
    elif "front_right" in image_result_dict["image_filename"]:
         image_result_dict["image_perspective"] = "front_right"
    elif "front_left" in image_result_dict["image_filename"]:
         image_result_dict["image_perspective"] = "front_left"
    elif "upper" in image_result_dict["image_filename"]:
         image_result_dict["image_perspective"] = "upper"
    elif "lower" in image_result_dict["image_filename"]:
         image_result_dict["image_perspective"] = "lower"

    i = 0 

    # categorize image annotation data based on id to be able to sweep all records related to each label id 
    image_annotation_data = categorize_by_id(json_image_record['annotations'][0]['result'])
    keys = list(image_annotation_data.keys())
    if verbos: print(image_result_dict["image_filename"])
    #useful record data will be collected here
    for key in keys:
          #print(key)
          #container for collecting image annotation data
          tooth_result_dict = {
                    "tooth_number": 0,
                    "region_class": "",
                    "label_id": "",
                    "missing" : "",
                    "shape_attributes":{}
                    }
          for record in image_annotation_data[key]:
               if record['from_name'] == "labels": # "labels" is used for rectangle shape annotations
                    if not test_set: tooth_result_dict["shape_attributes"] = {'x': record['value']['x'],
                                                       'y': record['value']['y'],
                                                       'width': record['value']['width'] ,
                                                       'height': record['value']['height'],
                                                       'rotation':record['value']['rotation'] ,
                                                       "shape": "rectangle"}
                    
               elif record['from_name'] == "labels2": # "labels2" is used for polygon shape annotations
                    [x,y] = list(np.array(record['value']['points']).transpose())
                    tooth_result_dict["label_id"] = record['id']
                    if not test_set: tooth_result_dict["shape_attributes"] = {'all_points_x': list(x), 'all_points_y': list(y),"shape": "polygon"}

               elif record['from_name'] == "Tooth number":
                    if not flag_image_data_recorded:
                         image_result_dict["image_width"] = record['original_width']
                         image_result_dict["image_height"]= record['original_height']
                         flag_image_data_recorded = True
                    tooth_result_dict["tooth_number"] = int(record['value']['text'][0])
                    if not tooth_result_dict["region_class"] == "Missing":
                         tooth_result_dict["region_class"] = "T"+ str(record['value']['text'][0])
                    tooth_result_dict["label_id"] = record['id']
                
                    output_data_valid = True # valid means there is tooth data in the file. so it should be appended to the master annotations dictionary
                
               elif  record['from_name'] == "image quality":
                    image_result_dict["image_quality"] = record['value']['choices'][0]
               
               elif record['from_name'] == "missing":
                    tooth_result_dict["missing"] = record['value']['choices'][0]
                    if record['value']['choices'][0] in ('A', 'M'): # 15.10.2024: I added 'M' later because some labelers mistakenly marked missing as 'M' 
                         tooth_result_dict["region_class"] = "Missing"
                    output_data_valid = True


          
          # converting rectangle labels to polygon
          # percent to pixel
          #print(tooth_result_dict['shape_attributes'])
          if not test_set:
               if 'shape' in tooth_result_dict['shape_attributes'].keys():
                    if tooth_result_dict['shape_attributes']['shape'] == 'rectangle':
                         x1 = tooth_result_dict['shape_attributes']['x']
                         x2 = x1 + (tooth_result_dict['shape_attributes']['width']*0.01*image_result_dict['image_width'])
                         x3 = x2  
                         x4 = x1 
                         y1 = tooth_result_dict['shape_attributes']['y']
                         y2 = y1
                         y3 = y2 + (tooth_result_dict['shape_attributes']['height']*0.01*image_result_dict['image_height'])
                         y4 = y3
                         x_all = [ int(x) for x in [x1, x2, x3, x4] ]
                         y_all = [ int(x) for x in [y1, y2, y3, y4] ]
                         tooth_result_dict['shape_attributes']['all_points_x'] = x_all
                         tooth_result_dict['shape_attributes']['all_points_y'] = y_all
                         tooth_result_dict['shape_attributes']['shape'] = 'polygon'
                    elif tooth_result_dict['shape_attributes']['shape'] == 'polygon':
                         points_y = [element * (0.01*image_result_dict['image_height']) for element in tooth_result_dict['shape_attributes']['all_points_y']] # converting to pixel
                         points_y = [ int(x) for x in points_y ]
                         points_x = [element * (0.01*image_result_dict['image_width']) for element in tooth_result_dict['shape_attributes']['all_points_x']] # converting to pixel
                         points_x = [ int(x) for x in points_x ]
                         tooth_result_dict['shape_attributes']['all_points_x'] = points_x
                         tooth_result_dict['shape_attributes']['all_points_y'] = points_y
          
          
          if not record['from_name'] == "image quality":
                    if not tooth_result_dict['tooth_number'] == 0:
                         if  'all_points_y' in tooth_result_dict['shape_attributes'].keys():# I added this check to handle and ignore rare cases that there is inconsistency in label data structure that makes the function fail to add shape attributes. (4 cases!)
                              image_result_dict["teeth_data"][i] = tooth_result_dict
                              i = i+1
          
          # if  'all_points_y' not in tooth_result_dict['shape_attributes'].keys():# I added this check to handle and ignore rare cases that there is inconsistency in label data structure that makes the function fail to add shape attributes. (4 cases!)
          #      output_data_valid = False


    return image_result_dict, output_data_valid
                


def polygon_json_crowler (json_files_dir = "./final polygon jsons/", json_output_dir_name="./", output_file_name = "annotations.json", label_studio_file_id_len = 9, create_file = True, test_set = False, verbos = True):
    # iterates through the json files in the folder and creates one json file that contains polygons for each image &| tooth. 

    JSON_FILES_DIR = os.path.abspath(json_files_dir)
    annotations = {}
    i = 0
    for json_file_name in fnmatch.filter(os.listdir(JSON_FILES_DIR), '*.json'):
        loaded_json = json.load(open(os.path.join(JSON_FILES_DIR, json_file_name)))
        for record in loaded_json:

            try:
                image_annotations, valid = json_record_tooth_polygon_fetch(record, label_studio_file_id_len, test_set, verbos)
                if valid:
                    annotations[i] = image_annotations
                    i = i+1
            except:
                print("file failed to fetch name and ignored: "+ json_file_name )
                return annotations


    if create_file:
        json_string = json.dumps(annotations, ensure_ascii=False, indent=4)
        with open(os.path.join(json_output_dir_name , output_file_name) , "w") as outfile:
            outfile.write(json_string)
    
    return annotations
  

def get_teeth_classes_dict(annotations_json_dir = "./" , json_file_name = "annotations.json"):
    """This function fetches classes from the introduced json file (json file is created by json_crowler()). 
    Then returns a dictionary containing class names as keys and class numbers as values.
    class values are simply based on the class names order in the dictionary and "Missing is always the last one".
    Notice that class int values are not necessarily constant for different datasets.
    
    annotations_json_dir = "./"
    json_file_name 
    """
    annotations = json.load(open(os.path.join(annotations_json_dir, json_file_name)))
    teeth_classes_set = set()
    teeth_classes_dict = {}

    for photo_key in annotations:
        for teeth_data_key in annotations[photo_key]["teeth_data"]:
            teeth_classes_set.add(annotations[photo_key]["teeth_data"][teeth_data_key]["region_class"])
    
    if "Missing" in teeth_classes_set:
        teeth_classes_set.remove("Missing")
        sorted_list = sorted(teeth_classes_set, key=lambda x: int(x[1:]))
        sorted_list.append("Missing")
    else:
        sorted_list = sorted(teeth_classes_set, key=lambda x: int(x[1:]))

    teeth_classes_dict = {key: i + 1 for i, key in enumerate(sorted_list)}


    return teeth_classes_dict

def categorize_by_id(json_image_annotation_result):
  """
  Reads a JSON file, categorizes sub-dictionaries based on their "id" field,
  and returns a dictionary with the structure {"id": {sub_dict1, sub_dict2, ...}}
  """
  data = {}
  for item in json_image_annotation_result:
    # Check if item has an "id" field
    if "id" in item:
      item_id = item["id"]
      # Create a list for sub-dictionaries with the same id if it doesn't exist
      if item_id not in data:
        data[item_id] = []
      data[item_id].append(item)
  return data


def get_substring_after_p5c(text):
  """Extracts the substring after the last "%5C" in the given text.

  Args:
      text: The string to extract the substring from.

  Returns:
      The substring after the last "%5C", or the entire string if there's no "%5C".
  """
  # Split the string by '%5C' (not decoded backslash)
  parts = text.rsplit("%5C", 1)

  # If there's only one part, there's no backslash
  # Get the substring after the last '%5C' (if it exists)
  substring = parts[-1] if len(parts) > 1 else text
  return substring




def split_dataset_by_filename_prefix(jsons_dir, image_dataset_dir, root_dir, seed=0, train_val_test_percent = [80, 10, 10] , num_id_letters = 7, verbose = False, remove_exif = True):
  """
  Splits files in a source directory based on first `num_id_letters` letters of filename into training and validation sets.
  Image files can have EXIF metadata in them. This metadata some times include orientation information. Not all softwares and libraries handle the data the same way. So we remove this info during copying
  image files to train, val and test folders. 

  Args:
      jsons_dir (str): Path to the source directory containing json files.
      image_dataset_dir (str): Path to the file source. in this case, it is the images folder.
      root_dir (str): Path to train, val and test root directory.
      seed (int): to reproduce the same splitting behavior. 
      train_val_test_percent (list): List containing three integer values representing the split percentages for training, validation and test data (e.g., [80, 10, 10]).
      num_id_letters (int): Number of the first letters of the filename to use for identification.
      remove_exif (bool): if True, removes EXIT metadata info from images.
      verbos (bool): if True, prints information about the process
      
  Raises:
      ValueError: If `train_val_percent` does not sum to 100 or contains negative values.
  """

  if not (sum(train_val_test_percent) == 100 and all(x >= 0 for x in train_val_test_percent)):
    raise ValueError("train_val_percent must sum to 100 and contain non-negative values.")

  # Create empty lists for training and validation file IDs
  train_ids = []
  val_ids = []
  test_ids = []

  # Get all filenames from the source and image directories
  filenames = [filename for filename in os.listdir(jsons_dir)]
  image_file_names = [filename for filename in os.listdir(image_dataset_dir)]


  # Split filenames randomly based on train_val_percent
  if seed !=0: random.seed(seed)
  random.shuffle(filenames)  # Randomly shuffle filenames before splitting
  num_files = len(filenames)
  train_split = int(num_files * train_val_test_percent[0] / 100)
  train_ids = filenames[:train_split]
  val_split = int(num_files * train_val_test_percent[1] / 100)
  if train_val_test_percent[1] != 0:
    val_ids = filenames[train_split:(train_split+val_split)]
    test_ids = filenames[(train_split+val_split):]
  else:
    val_ids = filenames[train_split:]
    test_ids = []
  train_ids = [item[:num_id_letters] for item in train_ids]
  val_ids = [item[:num_id_letters] for item in val_ids]
  test_ids = [item[:num_id_letters] for item in test_ids]
  #print(train_ids)
  #print(val_ids)

  # Create destination directories if they don't exist
  train_path = os.path.join(root_dir, "train")
  val_path = os.path.join(root_dir, "val")
  test_path = os.path.join(root_dir, "test")
  os.makedirs(train_path, exist_ok=True)
  if len(val_ids) != 0: os.makedirs(val_path, exist_ok=True)
  if len(test_ids) != 0: os.makedirs(test_path, exist_ok=True)

  # Copy files based on IDs
  for filename in filenames:
    file_id = filename[:num_id_letters]
    shortlist = [i for i in image_file_names if i.startswith(file_id)]
    
    ### Training set ########################################################
    if verbose: print(filename)
    if file_id in train_ids:
      source_path = os.path.join(jsons_dir, filename)
      try:
        copy(source_path, train_path)
      except shutil.SameFileError as e:
        print(e)
        pass
        
      # Find and copy the corresponding file from image_dataset_dir
      for potential_image in shortlist:
          image_path = os.path.join(image_dataset_dir, potential_image)
          if verbose: print(potential_image)
          if remove_exif:
            img = cv2.imread(image_path)  # OpenCV ignores EXIF metadata while reading.
            cv2.imwrite(os.path.join(train_path, potential_image), img)
          else:
            try:
              copy(image_path, train_path)
            except shutil.SameFileError as e:
              print(e)
              pass
          
    ### Validation set ########################################################
    elif file_id in val_ids:
      source_path = os.path.join(jsons_dir, filename)
      try:
        copy(source_path, val_path)
      except shutil.SameFileError as e:
        print(e)
        pass

      # Find and copy the corresponding file from image_dataset_dir
      for potential_image in shortlist:
          image_path = os.path.join(image_dataset_dir, potential_image)
          if verbose: print(potential_image)
          if remove_exif:
            img = cv2.imread(image_path)  # OpenCV ignores EXIF metadata while reading.
            cv2.imwrite(os.path.join(val_path, potential_image), img)
          else:
            try:
              copy(image_path, val_path)
            except shutil.SameFileError as e:
              print(e)
              pass

    ### Test set ########################################################
    elif file_id in test_ids:
      source_path = os.path.join(jsons_dir, filename)
      try:
        copy(source_path, test_path)
      except shutil.SameFileError as e:
        print(e)
        pass

      # Find and copy the corresponding file from image_dataset_dir
      for potential_image in shortlist:
          image_path = os.path.join(image_dataset_dir, potential_image)
          if verbose: print(potential_image)
          if remove_exif:
            img = cv2.imread(image_path)  # OpenCV ignores EXIF metadata while reading.
            cv2.imwrite(os.path.join(test_path, potential_image), img)
          else:
            try:
              copy(image_path, test_path)
            except shutil.SameFileError as e:
              print(e)
              pass


  print(f"Training data copied to: {train_path}")
  if len(val_ids) != 0: print(f"Validation data copied to: {val_path}")
  if len(test_ids) != 0: print(f"Test data copied to: {test_path}")



  # ###############################################
  # Semantic data preprocessing tools
  # ###############################################


def json_record_tooth_semantic_fetch(json_image_record, redundant_id_len, test_set = False, verbos = False):
     """fetch tooth **semantic** label data from image record in the json file.
      this function should be called from another loop that iterates all image records in the json file.

      json_image_record: record fetched from the json file. based on the known structure, each record has all the data for one image.
      images_dir: images path. function uses this path to retrieve images and their shape info because this info is not provided in the json file. 
                    DL models like mrcnn need this info to prepare image and labels for training.
      redundant_id_len: label studio creates random id for each uploaded file. we need to remove this id from file name to make it compatible with local file names.
     """
     image_result_dict = {"image_filename":"", "image_perspective": "", "image_width": 0, "image_height": 0, "label_data": {} }
     label_data = {}
     output_data_valid = False # i fthe file contain valid data, this flag will rise.
     flag_image_data_recorded = False # to prevent overwriting image size data

     #get file name. some json files in the dataset have slightly different format. So, I use try except to handle them. if failed raises the except to the caller func.
     try:
          image_result_dict["image_filename"] = json_image_record['file_upload'][redundant_id_len:]
     except:
          try:
               image_result_dict["image_filename"] = get_substring_after_p5c(json_image_record['data']['image'])
          except:
               raise
          #find out the camera perspective
     if "front-" in image_result_dict["image_filename"]:
          image_result_dict["image_perspective"] = "front"
     elif "front_right" in image_result_dict["image_filename"]:
          image_result_dict["image_perspective"] = "front_right"
     elif "front_left" in image_result_dict["image_filename"]:
          image_result_dict["image_perspective"] = "front_left"
     elif "upper" in image_result_dict["image_filename"]:
          image_result_dict["image_perspective"] = "upper"
     elif "lower" in image_result_dict["image_filename"]:
          image_result_dict["image_perspective"] = "lower"
     #image shape
     #image_result_dict['image_width'], image_result_dict['image_height'] = get_image_shape(images_dir, image_result_dict['image_filename'])

     # categorize image annotation data based on id to be able to sweep all records related to each label id 
     ###########################Some of the records do not contain result field or do not contain annotation data under "result" field. if this is the case, function will return from here.
     if 'result' not in json_image_record['annotations'][0] or not json_image_record['annotations'][0]['result']:
          image_file_name=image_result_dict["image_filename"]
          message = f"image '{image_file_name}' did not have annotations."
          if verbos: print(message)
          return message, False
     ###########################
     image_annotation_data = categorize_by_id(json_image_record['annotations'][0]['result'])
     keys = list(image_annotation_data.keys())
     
     #useful record data will be collected here
     i = 0 
     for key in keys: # each key refers to one image  
          semantic_label = {"tooth_number": 0, "region_class": "", "type": 0,"tooth_surface": [], "label_id": "", "rle":[]}

          # This for loop collects related data from several annotation records and records them in semantic_label
          for record in image_annotation_data[key]: # each key refers to one annotation record. each label has several annotation records.

               if record['from_name'] == "tag": # "tag" is used for brushable rle annotations
                    if not test_set: semantic_label["rle"] = record['value']['rle']
                    if not flag_image_data_recorded:
                         image_result_dict["image_width"] = record['original_width']
                         image_result_dict["image_height"]= record['original_height']
                         flag_image_data_recorded = True
                    semantic_label["label_id"] = record['id']
                              
               elif record['from_name'] == "surface": # "labels2" is used for polygon shape annotations
                    semantic_label["tooth_surface"] = record['value']["choices"]

               elif record['from_name'] == "Tooth number":
                    semantic_label["tooth_number"] = int(record['value']['text'][0])
               
               
               # labels of caries are irregularly annotated.
               elif record['from_name']=="Gingival recession":
                    semantic_label["region_class"] = "GiRe"
                    semantic_label["type"] = int(record['value']['choices'][0])
                    output_data_valid = True # valid means there is tooth data in the file. so it should be appended to the master annotations dictionary
               
               # labels of caries are irregularly annotated. The 'from_name' value for caries is "caries: severity of observed caries"
               elif record['from_name'] in ["Other", "other", "Caries", "caries","caries: severity of observed caries", "Hypoplasia", "Filling", "Plaque", "Calculus", "Erosion", "Gingivitis", "Gingival recession"]:
                    semantic_label["region_class"] = record['from_name'][:3]
                    semantic_label["type"] = int(record['value']['choices'][0])
                    output_data_valid = True # valid means there is tooth data in the file. so it should be appended to the master annotations dictionary


          label_data.update({semantic_label['label_id']:semantic_label})
          i = i+1
          
     image_result_dict['label_data'] = label_data

     return image_result_dict, output_data_valid

#I don't need to create a json level for teeth. put all lables in a key named labels


def semantic_json_crowler (json_files_dir = "./final semantic jsons/", json_output_dir_name="./", output_filename = "semantic_annotations.json", label_studio_file_id_len = 9, create_file = True, test_set = False, verbos = False):
    # iterates through the json files in the folder and creates one json file that contains polygons for each image &| tooth. 

    JSON_FILES_DIR = os.path.abspath(json_files_dir)
    annotations = {}
    i = 0
    for json_file_name in fnmatch.filter(os.listdir(JSON_FILES_DIR), '*.json'):
        loaded_json = json.load(open(os.path.join(JSON_FILES_DIR, json_file_name)))
        j = 0
        for record in loaded_json:
            try:
                if verbos: print(f"Started to fetch image {j} data from json file: {json_file_name}")
                image_annotations, valid = json_record_tooth_semantic_fetch(record, label_studio_file_id_len, test_set, verbos= verbos)
                j = j+1
                if valid:
                    annotations[i] = image_annotations
                    i = i+1
            except:
                print("file failed to fetch name and ignored: "+ json_file_name )
                return annotations


    if create_file:
        json_string = json.dumps(annotations, ensure_ascii=False, indent=4)
        with open(os.path.join(json_output_dir_name, output_filename) , "w") as outfile:
            outfile.write(json_string)
            if verbos: print("semantic_annotations.json file created.")
    
    return annotations




def get_teeth_semantic_classes_dict(annotations_json_dir = "./" , json_file_name = "semantic_annotations.json"):
    """This function fetches classes from the introduced json file (json file is created by semantic_json_crowler()). 
    Then returns a dictionary containing class names as keys and class numbers as values.
    class values are simply based on the class names order in the dictionary and "Missing is always the last one".
    Notice that class int values are not necessarily constant for different datasets.
    
    annotations_json_dir = "./"
    json_file_name 
    """
    annotations = json.load(open(os.path.join(annotations_json_dir, json_file_name)))
    teeth_semantic_classes_set = set()

    for photo_key in annotations:
        for label_data_key in annotations[photo_key]["label_data"]:
            teeth_semantic_classes_set.add(annotations[photo_key]["label_data"][label_data_key]["region_class"])
    
    sorted_list = sorted(teeth_semantic_classes_set)

    teeth_classes_dict = {key: i + 1 for i, key in enumerate(sorted_list)}

    return teeth_classes_dict

def get_toothwise_semantic_classes_dict(annotations_json_dir = "./" , json_file_name = "toothwise_annotations.json"):
    """This function fetches classes from the introduced json file (json file is created by semantic_json_crowler()). 
    Then returns a dictionary containing class names as keys and class numbers as values.
    class values are simply based on the class names order in the dictionary and "Missing is always the last one".
    Notice that class int values are not necessarily constant for different datasets.
    
    annotations_json_dir = "./"
    json_file_name 
    """
    annotations = json.load(open(os.path.join(annotations_json_dir, json_file_name)))
    teeth_semantic_classes_set = set()

    for photo_key in annotations:
        # print(photo_key)
        for tooth_data_key in annotations[str(photo_key)]["teeth_data"]:
            # print(tooth_data_key)
            if annotations[str(photo_key)]["teeth_data"][str(tooth_data_key)]['diagnostic_labels']:
                # print("ok")
                for semantic_key in annotations[str(photo_key)]["teeth_data"][str(tooth_data_key)]['diagnostic_labels']:
                    
                    teeth_semantic_classes_set.add(annotations[str(photo_key)]["teeth_data"][str(tooth_data_key)]['diagnostic_labels'][semantic_key]['region_class'])
    
    sorted_list = sorted(teeth_semantic_classes_set)

    teeth_classes_dict = {key: i + 1 for i, key in enumerate(sorted_list)}

    return teeth_classes_dict

# #
# # functions for creating semantic datasets with subclasses
# #

# def filter_dictionary(dictionary, target_key , target_values):
#     """Filters a multi-level dictionary based on a key-value pair in the third level.

#     Args:
#         dictionary: The input dictionary.
#         target_value: The target value to filter for.

#     Returns:
#         A new dictionary containing only the key-value pairs that match the target condition.
#     """

#     filtered_dict = {}
#     for key1, value1 in dictionary.items():
#         if isinstance(value1, dict) and value1[target_key] in target_values:
#                     filtered_dict[key1] = value1
#     return filtered_dict


# def filter_classes_semantic_json(file_path, target_classes_list):
#     """Filters a multi-level dictionary based on a key-value pair in the third level.

#     Args:
#         file_path: file path ro semantic annotations json file.
#         target_classes_list: The list contains target classes to filter for.

#     Returns:
#         A new dictionary containing only the records that contain the target classes.
#     """
#     loaded_json = json.load(open(file_path))
#     target_dict = {}
#     target_label_record = {}
#     i = 0
#     for key, value in loaded_json.items():
#         target_label_record = filter_dictionary(value['label_data'], target_key='region_class', target_values= target_classes_list )
#         if target_label_record: 
#             target_dict[i] = value
#             target_dict[i]['label_data'] = target_label_record
#             i = i+1
#     return target_dict





#
# filtering semantic datasets based on class subset list
#

def filter_dictionary(dictionary, target_key , target_values):
    """Filters a multi-level dictionary based on a key-value pair in the third level.

    Args:
        dictionary: The input dictionary.
        target_value: The target value to filter for.

    Returns:
        A new dictionary containing only the key-value pairs that match the target condition.
    """

    filtered_dict = {}
    for key1, value1 in dictionary.items():
        if isinstance(value1, dict) and value1[target_key] in target_values:
                    filtered_dict[key1] = value1
    return filtered_dict


def filter_classes_semantic_json(file_path, target_classes_list):
    """Filters a multi-level dictionary based on a key-value pair in the third level.

    Args:
        file_path: file path ro semantic annotations json file.
        target_classes_list: The list contains target classes to filter for.

    Returns:
        A new dictionary containing only the records that contain the target classes.
    """
    loaded_json = json.load(open(os.path.join(file_path, 'semantic_annotations.json')))
    target_dict = {}
    target_label_record = {}
    i = 0
    for key, value in loaded_json.items():
        target_label_record = filter_dictionary(value['label_data'], target_key='region_class', target_values= target_classes_list )
        if target_label_record: 
            target_dict[i] = value
            target_dict[i]['label_data'] = target_label_record
            i = i+1
    print('number of images: ', i)
    return target_dict


def create_filtered_semantic_dataset(source_file_path, target_classes_list, output_filename, json_output_dir_name = './', create_file = False, verbose = True):
    annotations = filter_classes_semantic_json(source_file_path, target_classes_list)
    if create_file:
            json_string = json.dumps(annotations, ensure_ascii=False, indent=4)
            with open(os.path.join(json_output_dir_name, output_filename) , "w") as outfile:
                outfile.write(json_string)
                if verbose: print(f"{output_filename} file created in {json_output_dir_name}")
    return True