import os
import pickle
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import PIL
import wandb
import imageio

from preprocess.dataset_util import FrameDetections, sample_action_anticipation_frames, fetch_data, save_video_info
from preprocess.traj_util import compute_hand_traj
from preprocess.obj_util import compute_obj_traj
from preprocess.affordance_util import compute_obj_affordance
from preprocess.vis_util import vis_affordance, vis_hand_traj, vis_obj_traj

from hoa.visualisation import DetectionRenderer
from hoa.io import load_detections

def create_dummy_output_video(frames_path, frames_idxs, save_path, video_id, start_frame=None, hand_side=None):
    imgio_kargs = {'fps': 30, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    
    if start_frame is not None:
        output_video_file = save_path + f'/{video_id}_{start_frame}_{hand_side}.mp4'
    else:
        output_video_file = save_path + f'/{video_id}_{hand_side}.mp4'
    writer = imageio.get_writer(output_video_file, **imgio_kargs)  
    const_img = np.zeros((256, 456, 3), dtype=np.uint8)
    for frame_idx in frames_idxs:  
        frame = cv2.imread(os.path.join(frames_path, "frame_{:010d}.jpg".format(frame_idx)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        # print("shape of frameeeeeeeeee: ", frame.shape, const_img.shape)

        # Concatenate the first image and the current image side-by-side
        concatenated_image = cv2.hconcat([const_img, frame])
        writer.append_data(concatenated_image)
    writer.close()
    return output_video_file

def create_output_video(frames_path, frames_idxs, const_img, save_path, video_id, start_frame=None, hand_side=None):
    imgio_kargs = {'fps': 30, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    
    if start_frame is not None:
        output_video_file = save_path + f'/{video_id}_{start_frame}_{hand_side}.mp4'
    else:
        output_video_file = save_path + f'/{video_id}_{hand_side}.mp4'
    writer = imageio.get_writer(output_video_file, **imgio_kargs)  
    const_img = cv2.cvtColor(const_img, cv2.COLOR_BGR2RGB) 
    for frame_idx in frames_idxs:  
        frame = cv2.imread(os.path.join(frames_path, "frame_{:010d}.jpg".format(frame_idx)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

        # Concatenate the first image and the current image side-by-side
        concatenated_image = cv2.hconcat([const_img, frame])
        writer.append_data(concatenated_image)
    writer.close()
    return output_video_file

def create_task_video(frames_path, frame_idxs, save_path, video_id):
    height, width = 256, 456
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_file = save_path + f'/{video_id}.mp4'
    output_video = cv2.VideoWriter(output_video_file, fourcc, 30, (width, height))
    for frame_idx in frame_idxs:  # Adjust the range as per the number of images you have
        frame = cv2.imread(os.path.join(frames_path, "frame_{:010d}.jpg".format(frame_idx)))
        # Write the concatenated frame to the output video
        output_video.write(frame)

def save_segment(row):
    images_directory = f"/home/arpit/2g1n6qdydwa9u22shpxqzp0t8m/{row['participant_id']}/rgb_frames/{row['video_id']}"
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

# def create_output_video(frames_path, frame_idxs, const_img, save_path):
#     height, width, layers = const_img.shape
#     print("height, width: ", height, width)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     output_video_file = save_path + '/output_video.mp4'
#     output_video = cv2.VideoWriter(output_video_file, fourcc, 30, (2 * width, height))
#     for frame_idx in frame_idxs:  # Adjust the range as per the number of images you have
#         frame = cv2.imread(os.path.join(frames_path, "frame_{:010d}.jpg".format(frame_idx)))
        
#         # Concatenate the first image and the current image side-by-side
#         concatenated_image = cv2.hconcat([const_img, frame])
        
#         # Write the concatenated frame to the output video
#         output_video.write(concatenated_image)

wandb_log = False
save_video = False
contact_frames_table = None
if wandb_log:
    wandb.login()
    run = wandb.init(
        project="contact-affordance",
        notes="trial experiment",
        tags=["baseline", "paper1"],
    )
    contact_frames_table = wandb.Table(columns=["Video Name", "Left Hand Contact Info", "Right Hand Contact Info"])
    contact_points_table = wandb.Table(columns=["Video Name", "Left Hand Contact Points", "Right Hand Contact Points"])
# run.log({"table_key": contact_frames_table})

train_df = pd.read_csv('assets/EPIC_100_train.csv')
with open('assets/open_obj_indices.pkl', 'rb') as f:
    idxs = pickle.load(f)
df = train_df.iloc[idxs]
print("size of filtered dataset: ", df.shape)

downloaded_participants = ['P01', 'P02', 'P03', 'P04', 'P06', 'P07', 'P09', 'P11', 'P12', 'P22', 'P23', 'P25', 'P26', 'P27', 'P28', 'P30', 'P33' 'P34', 'P35', 'P36', 'P37']
useless_videos = 0
for i in range(df.shape[0]):
# for i in range(300, 600):
    if df.iloc[i]['participant_id'] not in downloaded_participants:
        continue
    images_directory = f"/home/arpit/2g1n6qdydwa9u22shpxqzp0t8m/{df.iloc[i]['participant_id']}/rgb_frames/{df.iloc[i]['video_id']}"
    if not os.path.exists(images_directory):
        continue
    # remove later
    # if df.iloc[i]['video_id'] != 'P01_105' or df.iloc[i]['start_frame'] != 13124:
    #     continue
    if df.iloc[i]['video_id'] != 'P02_129' or df.iloc[i]['start_frame'] != 17007:
        continue
    
    participant_id = df.iloc[i]['participant_id']
    video_id = df.iloc[i]['video_id']
    start_frame, end_frame = max(0, df.iloc[i]['start_frame']-60), df.iloc[i]['stop_frame'] # -60 so that we can leave some frames before the actual start of the action 


    frames_path = os.path.join('/home/arpit/2g1n6qdydwa9u22shpxqzp0t8m', participant_id, "rgb_frames", video_id + "/")
    ho_path = os.path.join('/home/arpit/2g1n6qdydwa9u22shpxqzp0t8m/', participant_id, "hand-objects", "{}.pkl".format(video_id))

    frames_idxs = np.arange(start_frame, end_frame, dtype=int).tolist()
    print("-----------------------------------------------")
    print("p_id, video_id, start_frame, end_frame: ", participant_id, video_id, start_frame, end_frame)
    print("total frames that we will be looking at: ", len(frames_idxs))

    save_path = 'arpit_output'
    # save_segment(df.iloc[i])

    with open(ho_path, "rb") as f:
        video_detections = [FrameDetections.from_protobuf_str(s) for s in pickle.load(f)]
    print("ho_detections: ", video_detections[50])

    retval = fetch_data(frames_path, video_detections, frames_idxs, contact_frames_table, wandb_log=wandb_log, save_video=save_video)
    if wandb_log:
        new_contact_frames_table = wandb.Table(
            columns=contact_frames_table.columns, data=contact_frames_table.data
        )
        run.log({"contact frames": new_contact_frames_table})  
    
    if retval is None:
        print("Failed to find contact frame for: ", participant_id, video_id, start_frame, end_frame)
        continue
    frames_idxs_left, frames_left, annots_left, frames_idxs_right, frames_right, annots_right = retval
    # print("Lengths: frame_idxs, frames, annots: ", len(frames_idxs), len(frames), len(annots))

    print("len for left and right hand frmaes: ", len(frames_left), len(frames_right))
    left_hand_contact_points, right_hand_contact_points = False, False
    output_video_file_left, output_video_file_right = None, None
    # ----------------- Compute affordance for left hand -----------------------
    results_hand_left = compute_hand_traj(frames_left, annots_left, ['LEFT'], hand_threshold=0.5, obj_threshold=0.5)
    if results_hand_left is None:
        print("compute traj failed in main for the left hand")  # homography fails or not enough points
    else:
        homography_stack_left, hand_trajs_left = results_hand_left
        # print("hand_trajs: ", hand_trajs_left.keys())
        # ------ visualize hand trajectory ---------
        img_vis = vis_hand_traj(frames_left, hand_trajs_left)
        # save image 
        img = cv2.hconcat([img_vis, frames_left[-1]])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        save_path_hand_traj = save_path + '/hand_trajectory'
        cv2.imwrite(os.path.join(save_path_hand_traj, "left_hand_traj_{}_{}.jpg".format(video_id, start_frame)), img)
        # ------------------------------------------
        results_obj_left = compute_obj_traj(frames_left, annots_left, ['LEFT'], homography_stack_left,
                                    hand_threshold=0.5,
                                    obj_threshold=0.5,
                                    contact_ratio=0.4)
        if results_obj_left is None:
            print("compute obj traj failed in main")
        else:
            contacts, obj_trajs, active_obj, active_object_idx, obj_bboxs_traj = results_obj_left
            # print("active_obj: ", active_obj)
            # print("active_object_idx: ", active_object_idx)
            # print("obj_trajs.keys(): ", obj_trajs.keys())
            # ------ visualize object trajectory ---------
            img_vis = vis_obj_traj(frames_left, obj_trajs)
            # save image 
            save_path_obj_traj = save_path + '/object_trajectory'
            img = cv2.hconcat([img_vis, frames_left[-1]])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            cv2.imwrite(os.path.join(save_path_obj_traj, "left_obj_traj_{}_{}.jpg".format(video_id, start_frame)), img)
            #-------------------------------------------
            frame, annot, homography = frames_left[-1], annots_left[-1], homography_stack_left[-1]

            affordance_info = compute_obj_affordance(frame, annot, active_obj, active_object_idx, homography,
                                                                active_obj_traj=obj_trajs['traj'], obj_bboxs_traj=obj_bboxs_traj,
                                                                num_points=10, num_sampling=20, start_frame=frames_left[0], hand_side='LEFT')
            
            # print("affordance_info: ", affordance_info)
            if affordance_info is not None:
                left_hand_contact_points = True
                if save_video:
                    img_pts, img_pts_homo, img_hmap = vis_affordance(frames_left[0], affordance_info, contact_frame=frames_left[-1])
                    img = cv2.hconcat([img_pts, img_pts_homo, img_hmap])
                    contact_points_save_path = save_path + '/contact_points'
                    output_video_file_left = create_output_video(frames_path, frames_idxs, img, contact_points_save_path, video_id, start_frame=frames_idxs_left[0], hand_side='LEFT')
                # cv2.imwrite(os.path.join(save_path, "contact_points/affordace_{}_{}_left.jpg".format(video_id, start_frame)), img)

    
    # ----------------- Compute affordance for right hand -----------------------
    results_hand_right = compute_hand_traj(frames_right, annots_right, ['RIGHT'], hand_threshold=0.5, obj_threshold=0.5)
    if results_hand_right is None:
        print("compute traj failed in main for the right hand")  # homography fails or not enough points
    else:
        homography_stack_right, hand_trajs_right = results_hand_right
        # print("hand_trajs: ", hand_trajs)
        # ------ visualize hand trajectory ---------
        img_vis = vis_hand_traj(frames_right, hand_trajs_right)
        # save image 
        img = cv2.hconcat([img_vis, frames_right[-1]])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        save_path_hand_traj = save_path + '/hand_trajectory'
        cv2.imwrite(os.path.join(save_path_hand_traj, "right_hand_traj_{}_{}.jpg".format(video_id, start_frame)), img)
        # ------------------------------------------
        results_obj_right = compute_obj_traj(frames_right, annots_right, ['RIGHT'], homography_stack_right,
                                    hand_threshold=0.5,
                                    obj_threshold=0.5,
                                    contact_ratio=0.4)
        if results_obj_right is None:
            print("compute obj traj failed")
        else:
            contacts, obj_trajs, active_obj, active_object_idx, obj_bboxs_traj = results_obj_right
            # print("active_obj: ", active_obj)
            # print("active_object_idx: ", active_object_idx)
            # ------ visualize object trajectory ---------
            img_vis = vis_obj_traj(frames_right, obj_trajs)
            # save image 
            img = cv2.hconcat([img_vis, frames_right[-1]])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            save_path_obj_traj = save_path + '/object_trajectory'
            cv2.imwrite(os.path.join(save_path_obj_traj, "right_obj_traj_{}_{}.jpg".format(video_id, start_frame)), img)
            #-------------------------------------------
            frame, annot, homography = frames_right[-1], annots_right[-1], homography_stack_right[-1]

            affordance_info = compute_obj_affordance(frame, annot, active_obj, active_object_idx, homography,
                                                                active_obj_traj=obj_trajs['traj'], obj_bboxs_traj=obj_bboxs_traj,
                                                                num_points=5, num_sampling=20, start_frame=frames_right[0], hand_side='RIGHT')
            
            # print("affordance_info: ", affordance_info)
            if affordance_info is not None:
                right_hand_contact_points = True
                if save_video:
                    img_pts, img_pts_homo, img_hmap = vis_affordance(frames_right[0], affordance_info, contact_frame=frames_right[-1])
                    img = cv2.hconcat([img_pts, img_pts_homo, img_hmap])
                    contact_points_save_path = save_path + '/contact_points'
                    output_video_file_right = create_output_video(frames_path, frames_idxs, img, contact_points_save_path, video_id, start_frame=frames_idxs_right[0], hand_side='RIGHT')

    
    if wandb_log:
        contact_points_save_path = save_path + '/contact_points'
        if output_video_file_left is None:
            output_video_file_left = create_dummy_output_video(frames_path, frames_idxs, contact_points_save_path, video_id, start_frame=frames_idxs_left[0], hand_side='LEFT')
        if output_video_file_right is None:
            output_video_file_right = create_dummy_output_video(frames_path, frames_idxs, contact_points_save_path, video_id, start_frame=frames_idxs_right[0], hand_side='RIGHT')
                   
        contact_points_table.add_data(f'{video_id}_{start_frame}.mp4\n [start_frame, contact_frame, video]',
                            wandb.Video(output_video_file_left, fps=30, format="mp4"),
                            wandb.Video(output_video_file_right, fps=30, format="mp4"))   
        new_contact_points_table = wandb.Table(
            columns=contact_points_table.columns, data=contact_points_table.data
        )
        run.log({"contact points": new_contact_points_table})
    print("Doneeee")  
