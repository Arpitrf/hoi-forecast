import pickle
import pandas as pd
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import os
# path_root = Path('/home/arpit/test_projects/ekhand/')
# sys.path.append(str(path_root))
# from types import FrameDetections

def create_task_video(frames_path, frame_idxs, save_path, video_id):
    height, width = 256, 456
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_file = save_path + f'/{video_id}.mp4'
    output_video = cv2.VideoWriter(output_video_file, fourcc, 30, (width, height))
    for frame_idx in frame_idxs:  # Adjust the range as per the number of images you have
        frame = cv2.imread(os.path.join(frames_path, "frame_{:010d}.jpg".format(frame_idx)))
        # Write the concatenated frame to the output video
        output_video.write(frame)

def save_segment(index, row, images_directory):
    start_frame = row['start_frame']
    end_frame = row['stop_frame']
    save_path = f'arpit_output/action_segments'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_file = save_path + f'/{row["video_id"]}_{row["start_frame"]}.mp4'
    print("output_video_file: ", output_video_file)
    output_video = cv2.VideoWriter(output_video_file, fourcc, 30, (456, 256))

    for idx in range(start_frame-60, end_frame):
        file_name = "{:010d}".format(idx) 
        image_file = images_directory + f"/frame_{file_name}.jpg"
        # print("image_file: ", image_file)
        img = cv2.imread(image_file)
        # plt.imshow(img)
        # plt.show()
        # cv2.imshow('Video', img)
        # cv2.waitKey(25)  
        output_video.write(img)


train_df = pd.read_csv('assets/EPIC_100_train.csv')
validation_df = pd.read_csv('assets/EPIC_100_validation.csv')
print("train_df: ", train_df.shape, train_df.columns)
print("validation_df: ", validation_df.shape)
print("-------------------")
unique_participants = train_df['participant_id'].unique()
unique_videos = train_df['video_id'].unique()
print("unique_participants, unique_videos: ", unique_participants.shape, unique_videos.shape)

# tasks = train_df['narration'].value_counts()
# print("tasks: ", tasks)
# tasks_lis = tasks.axes[0].tolist()
# for i in range(len(tasks_lis)):
#     if 'open' in tasks_lis[i]:
#         print("an open task: ", tasks_lis[i])
    

# print(tasks[25:50])
# print("len(tasks): ", tasks.shape)
# print("df.shape: ", df[['participant_id', 'video_id', 'narration', 'start_frame', 'stop_frame']])

# # ------------------------- getting indices of particular actions ------------------------
# # keywords = ['open box', 'open tupperware', 'open bottle' , 'open container' , 'open jar' , 'open can' , 'open milk']
# keywords = ['open']
# non_keywords = ['fridge', 'cupboard', 'drawer', 'dishwasher', 'microwave', 'lid']
# counter = 0
# action_indices = [] 
# for i in range(train_df.shape[0]):
#     # print("narration: ", train_df.loc[i].at['narration'])
#     for k in keywords:
#         if k in train_df.loc[i].at['narration']:
#             print("narration: ", train_df.loc[i].at['narration'])
#             action_indices.append(i)
#             counter += 1
# print("Total action segments: ", counter)
# print("action_indices: ", action_indices)
# with open('assets/open_obj_indices.pkl', 'wb') as f:
#    pickle.dump(action_indices, f)
# # -----------------------------------------------------------------------------------------

# ------------------
non_keywords = ['fridge', 'refrigerator', 'cabinet', 'cupboard', 'drawer', 'dishwasher', 'microwave', 'lid', 'freezer', 'bin', 'machine', 'compost', 'oven', 'trash', 'scissors', 'door']
counter = 0
action_indices = [] 
for i in range(train_df.shape[0]):
    # print("narration: ", train_df.loc[i].at['narration'])
    if 'open' in train_df.loc[i].at['verb']:
        objs = train_df.loc[i].at['all_nouns']
        objs = objs.strip("]['").replace(':', ', ').split(', ')
        # print("objs: ", objs, type(objs))
        flag = True
        for obj in objs:
            if obj in non_keywords:
                flag = False
                break
        if flag:
            print("narration: ", train_df.loc[i].at['narration'])
            action_indices.append(i)
            counter += 1
print("Total action segments: ", counter)
print("action_indices: ", action_indices)
with open('assets/open_obj_indices_larger.pkl', 'wb') as f:
   pickle.dump(action_indices, f)
# ------------------


# with open('assets/open_obj_indices.pkl', 'rb') as f:
#     idxs = pickle.load(f)
# df = train_df.iloc[idxs]
# print("size of filtered dataset: ", df.shape)
# print("-----",df.iloc[0]['participant_id'])

# downloaded_participants = ['P01', 'P02', 'P03', 'P04', 'P06', 'P07', 'P09', 'P11', 'P12', 'P22', 'P23', 'P25', 'P26', 'P27', 'P28', 'P30', 'P33' 'P34', 'P35', 'P36', 'P37']
# counter = 0
# for index, row in df.iterrows():
#     counter += 1
#     if row['participant_id'] not in downloaded_participants:
#         continue
#     images_directory = f"/home/arpit/2g1n6qdydwa9u22shpxqzp0t8m/{row['participant_id']}/rgb_frames/{row['video_id']}"
#     if not os.path.exists(images_directory):
#         continue
#     print("index, counter: ", index, counter, row['start_frame'])
#     # save_segment(index, row, images_directory)
#     # print("saved video number: ", counter)
# print(counter)
