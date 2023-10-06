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


save_path = 'arpit_output'
# os.makedirs(, exist_ok=True)

participant_id = 'P02'
video_id = 'P02_102'
frames_path = os.path.join('/home/arpit/EPIC-KITCHENS', participant_id, "rgb_frames", video_id + "/")
ho_path = os.path.join('/home/arpit/EPIC-KITCHENS', participant_id, "hand-objects", "{}.pkl".format(video_id))
# start_act_frame = 1280
# frames_idxs = sample_action_anticipation_frames(start_act_frame, fps=30)
# frames_idxs = [1280, 1290, 1300, 1310, 1320, 1330, 1340, 1350, 1360, 1370]
# open ziplock
# frames_idxs = [32360, 32370, 32380, 32390, 32400]
# spread butter
# frames_idxs = [31437, 31442, 31447, 31452, 31457, 31462, 31467, 31472, 31477]
frames_idxs = np.arange(263, 302, 5, dtype=int).tolist()
print("frames_idxs: ", frames_idxs)

# detections = load_detections('detections/P01_101.pkl')
# frames = LazyFrameLoader('frames/P01_101')

with open(ho_path, "rb") as f:
    video_detections = [FrameDetections.from_protobuf_str(s) for s in pickle.load(f)]
print("ho_detections: ", len(video_detections))
results = fetch_data(frames_path, video_detections, frames_idxs)

if results is None:
    print("data fetch failed")
else:
    frames_idxs, frames, annots, hand_sides = results

    print("frames_idxs, frames: ", frames_idxs, np.array(frames).shape, type(annots[0]), hand_sides)

    # ----------- Vis ------------
    renderer = DetectionRenderer(hand_threshold=0.5, object_threshold=0.5)
    frame_idx = 0
    print("---", annots[frame_idx].hands)
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
                img_vis = vis_affordance(img_vis, affordance_info)
                img = cv2.hconcat([img_vis, frames[-1]])
                cv2.imwrite(os.path.join(save_path, "demo_{}.jpg".format(participant_id)), img)
                save_video_info(save_path, participant_id, frames_idxs, homography_stack, contacts, hand_trajs, obj_trajs, affordance_info)
    print(f"result stored at {save_path}")
