import os
import pickle
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import PIL

from preprocess.dataset_util import FrameDetections, sample_action_anticipation_frames, fetch_data, save_video_info
from preprocess.traj_util import compute_hand_traj
from preprocess.obj_util import compute_obj_traj
from preprocess.affordance_util import compute_obj_affordance
from preprocess.vis_util import vis_affordance, vis_hand_traj

from hoa.visualisation import DetectionRenderer
from hoa.io import load_detections

# class LazyFrameLoader:
#     def __init__(self, path: Union[Path, str], frame_template: str = 'frame_{:010d}.jpg'):
#         self.path = Path(path)
#         self.frame_template = frame_template

#     def __getitem__(self, idx: int) -> PIL.Image.Image:
#         return PIL.Image.open(str(self.path / self.frame_template.format(idx + 1)))

def create_output_video(frames_path, start_frame, end_frame, const_img, save_path):
    height, width, layers = const_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_file = save_path + '/output_video.mp4'
    output_video = cv2.VideoWriter(output_video_file, fourcc, 30, (2 * width, height))
    for frame_idx in range(start_frame, end_frame):  # Adjust the range as per the number of images you have
        frame = cv2.imread(os.path.join(frames_path, "frame_{:010d}.jpg".format(frame_idx)))
        
        # Concatenate the first image and the current image side-by-side
        concatenated_image = cv2.hconcat([const_img, frame])
        
        # Write the concatenated frame to the output video
        output_video.write(concatenated_image)

save_path = 'arpit_output'
# os.makedirs(, exist_ok=True)

participant_id = 'P03'
video_id = 'P03_101'
frames_path = os.path.join('/home/arpit/EPIC-KITCHENS', participant_id, "rgb_frames", video_id + "/")
ho_path = os.path.join('/home/arpit/EPIC-KITCHENS', participant_id, "hand-objects", "{}.pkl".format(video_id))
# P02_102
# start_frame, end_frame = 5598, 5684
# start_frame, end_frame = 6392, 6590
# start_frame, end_frame = 8167, 8253
# start_frame, end_frame = 8478, 8564
# P03_101
# start_frame, end_frame = 5219, 5260
# start_frame, end_frame = 32394, 32492
# start_frame, end_frame = 32752, 33014
# start_frame, end_frame = 7067, 7627
start_frame, end_frame = 6475, 6862

frames_idxs = np.arange(start_frame, end_frame, 5, dtype=int).tolist()
print("Total frames: ", len(frames_idxs))

# detections = load_detections('detections/P01_101.pkl')
# frames = LazyFrameLoader('frames/P01_101')

with open(ho_path, "rb") as f:
    video_detections = [FrameDetections.from_protobuf_str(s) for s in pickle.load(f)]
# print("ho_detections: ", len(video_detections))
results = fetch_data(frames_path, video_detections, frames_idxs)

if results is None:
    print("data fetch failed")
else:
    frames_idxs, frames, annots, hand_sides = results

    # print("frames_idxs, frames: ", frames_idxs, np.array(frames).shape, type(annots[0]), hand_sides)

    # ----------- Vis ------------
    renderer = DetectionRenderer(hand_threshold=0.5, object_threshold=0.5)
    frame_idx = 0
    # print("---", annots[frame_idx].hands)
    # plt.imshow(frames[frame_idx])
    # plt.show()
    x = renderer.render_detections(PIL.Image.fromarray(frames[frame_idx]), annots[frame_idx])
    # print("type(x): ", type(x))
    x.show()
    # ----------------------------
    # print("annots.shape: ", annots)
    # print("frames: ", np.array(frames).shape)
    # np_frames = np.array(frames)
    # fig, ax = plt.subplots(2,3)
    # ax[0, 0].imshow(np_frames[0])
    # ax[0, 1].imshow(np_frames[1])
    # ax[0, 2].imshow(np_frames[2])
    # ax[1, 0].imshow(np_frames[3])
    # ax[1, 1].imshow(np_frames[4])
    # plt.show()

    results_hand = compute_hand_traj(frames, annots, hand_sides, hand_threshold=0.1, obj_threshold=0.1)
    # just saving hand traj for now
    if results_hand is None:
        print("compute traj failed in main")  # homography fails or not enough points
    else:
        homography_stack, hand_trajs = results_hand
        img_vis = vis_hand_traj(frames, hand_trajs)
        # save image 
        img = cv2.hconcat([img_vis, frames[-1]])
        cv2.imwrite(os.path.join(save_path, "demo_{}.jpg".format(participant_id)), img)

        # save video
        img = create_output_video(frames_path, start_frame, end_frame, img_vis, save_path)

    if results_hand is None:
        print("compute traj failed in main")  # homography fails or not enough points
    else:
        homography_stack, hand_trajs = results_hand
        results_obj = compute_obj_traj(frames, annots, hand_sides, homography_stack,
                                        hand_threshold=0.1,
                                        obj_threshold=0.1,
                                        contact_ratio=0.4)
        if results_obj is None:
            print("compute obj traj failed")
        else:
            contacts, obj_trajs, active_obj, active_object_idx, obj_bboxs_traj = results_obj
            frame, homography = frames[-1], homography_stack[-1]
            # I think the affordance is being calculated based on the last frame which is not what we want
            affordance_info = compute_obj_affordance(frame, annots[-1], active_obj, active_object_idx, homography,
                                                         active_obj_traj=obj_trajs['traj'], obj_bboxs_traj=obj_bboxs_traj,
                                                         num_points=5, num_sampling=20)
            print("affordance_info: ", affordance_info)
            if affordance_info is not None:
                img_vis = vis_hand_traj(frames, hand_trajs)
                # img_vis = vis_hand_traj(frames, obj_trajs)
                # img_vis = vis_affordance(img_vis, affordance_info)
                img = cv2.hconcat([img_vis, frames[-1]])
                cv2.imwrite(os.path.join(save_path, "demo_{}.jpg".format(participant_id)), img)

                # concat image with GIF
                img = create_output_video(frames_path, start_frame, end_frame, img_vis, save_path)

                save_video_info(save_path, participant_id, frames_idxs, homography_stack, contacts, hand_trajs, obj_trajs, affordance_info)
    print(f"result stored at {save_path}")
