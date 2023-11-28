import os
import cv2
import numpy as np
from preprocess.ho_types import FrameDetections, HandDetection, HandSide, HandState, ObjectDetection
from hoa.visualisation import DetectionRenderer
import PIL
import matplotlib.pyplot as plt
import wandb
import imageio
    
def create_output_video(frames_path, frame_idxs, start_frame_img, contact_frame_img, save_path, video_id, start_frame=None, wandb_table=None, hand_side=None):
    imgio_kargs = {'fps': 30, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    
    if start_frame is not None:
        output_video_file = save_path + f'/{video_id}_{start_frame}_{hand_side}.mp4'
    else:
        output_video_file = save_path + f'/{video_id}_{hand_side}.mp4'
    writer = imageio.get_writer(output_video_file, **imgio_kargs)
    start_frame_img = cv2.cvtColor(start_frame_img.copy(), cv2.COLOR_BGR2RGB) 
    contact_frame_img = cv2.cvtColor(contact_frame_img.copy(), cv2.COLOR_BGR2RGB)   
    for frame_idx in frame_idxs:  
        frame = cv2.imread(os.path.join(frames_path, "frame_{:010d}.jpg".format(frame_idx)))
        frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB) 
        # plt.imshow(start_frame_img)
        # plt.show()
        # Concatenate the first image and the current image side-by-side
        concatenated_image = cv2.hconcat([start_frame_img, contact_frame_img, frame])
        writer.append_data(concatenated_image)
    writer.close()
    return output_video_file

    # print("wandb table data: ", wandb_table.data)
    # run.log({"table_key": wandb_table})
    # test_data_at.add(wandb_table, "predictions")
    # wandb.run.log_artifact(test_data_at)   

def sample_action_anticipation_frames(frame_start, t_buffer=1, fps=4.0, fps_init=60.0):
    time_start = (frame_start - 1) / fps_init
    num_frames = int(np.floor(t_buffer * fps))
    times = (np.arange(num_frames + 1) - num_frames) / fps + time_start
    times = np.clip(times, 0, np.inf)
    times = times.astype(np.float32)
    frames_idxs = np.floor(times * fps_init).astype(np.int32) + 1
    if frames_idxs.max() >= 1:
        frames_idxs[frames_idxs < 1] = frames_idxs[frames_idxs >= 1].min()
    return list(frames_idxs)


def load_ho_annot(video_detections, frame_index, imgW=456, imgH=256):
    annot = video_detections[frame_index-1] # frame_index start from 1
    assert annot.frame_number == frame_index, "wrong frame index"
    # print("((()))): ", annot)
    annot.scale(width_factor=imgW, height_factor=imgH)
    return annot


def load_img(frames_path, frame_index):
    frame = cv2.imread(os.path.join(frames_path, "frame_{:010d}.jpg".format(frame_index)))
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    return frame


def get_mask(frame, annot, hand_threshold=0.1, obj_threshold=0.1):
    msk_img = np.ones((frame.shape[:2]), dtype=frame.dtype)
    hands = [hand for hand in annot.hands if hand.score >= hand_threshold]
    objs = [obj for obj in annot.objects if obj.score >= obj_threshold]
    for hand in hands:
        (x1, y1), (x2, y2) = hand.bbox.coords_int
        msk_img[y1:y2, x1:x2] = 0

    if len(objs) > 0:
        hand_object_idx_correspondences = annot.get_hand_object_interactions(object_threshold=obj_threshold,
                                                                             hand_threshold=hand_threshold)
        for hand_idx, object_idx in hand_object_idx_correspondences.items():
            hand = annot.hands[hand_idx]
            object = annot.objects[object_idx]
            if not hand.state.value == HandState.STATIONARY_OBJECT.value:
                (x1, y1), (x2, y2) = object.bbox.coords_int
                msk_img[y1:y2, x1:x2] = 0
    return msk_img


def bbox_inter(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return xA, yA, xB, yB, 0

    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return xA, yA, xB, yB, iou


def compute_iou(boxA, boxB):
    boxA = np.array(boxA).reshape(-1)
    boxB = np.array(boxB).reshape(-1)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def points_in_bbox(point, bbox):
    (x1, y1), (x2, y2) = bbox
    (x, y) = point
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def valid_point(point, imgW=456, imgH=256):
    if point is None:
        return False
    else:
        x, y = point
        return (0 <= x < imgW) and (0 <=y < imgH)


def valid_traj(traj, imgW=456, imgH=256):
    if len(traj) > 0:
        num_outlier = np.sum([not valid_point(point, imgW=imgW, imgH=imgH)
                              for point in traj if point is not None])
        valid_ratio = np.sum([valid_point(point, imgW=imgW, imgH=imgH) for point in traj[1:]]) / len(traj[1:])
        valid_last = valid_point(traj[-1], imgW=imgW, imgH=imgH)
        if num_outlier > 1 or valid_ratio < 0.5 or not valid_last:
            traj = []
    return traj


def get_valid_traj(traj, imgW=456, imgH=256):
    try:
        traj[traj < 0] = traj[traj >= 0].min()
    except:
        traj[traj < 0] = 0
    try:
        traj[:, 0][traj[:, 0] >= imgW] = imgW - 1
    except:
        traj[:, 0][traj[:, 0] >= imgW] = imgW - 1
    try:
        traj[:, 1][traj[:, 1] >= imgH] = imgH - 1
    except:
        traj[:, 1][traj[:, 1] >= imgH] = imgH - 1
    return traj

def get_hand_object(annot, hand_threshold, obj_threshold, hand_side):
    hand_object_idx_correspondences = annot.get_hand_object_interactions(
                                        object_threshold=obj_threshold, hand_threshold=hand_threshold)
    hand, obj, obj_bbox = None, None, None
    for k, v in hand_object_idx_correspondences.items():
        # print(annot.hands[k].side.name)
        if annot.hands[k].side.name == hand_side:
            hand = annot.hands[k]
            obj = annot.objects[v]
            obj_bbox = obj.bbox
    return hand, obj, obj_bbox

def calculate_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    boxA = [pred_box.left, pred_box.top, pred_box.right, pred_box.bottom]
    boxB = [gt_box.left, gt_box.top, gt_box.right, gt_box.bottom]
    
    # 1. get the coordinate of inters
    ixmin = max(boxA[0], boxB[0])
    ixmax = min(boxA[2], boxB[2])
    iymin = max(boxA[1], boxB[1])
    iymax = min(boxA[3], boxB[3])

    iw = np.maximum(ixmax-ixmin, 0.)
    ih = np.maximum(iymax-iymin, 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((boxA[2]-boxA[0]) * (boxA[3]-boxA[1]) +
           (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni
    print("IOU: ", iou)
    return iou, inters

def calculate_iou_sizes(rect1, rect2):
    """
    Calculate Intersection over Union (IoU) between two rectangles of equal dimensions.

    Args:
    - rect1, rect2: Each rectangle should be a tuple (width, height) representing the dimensions of the rectangle.

    Returns:
    - IoU: Intersection over Union value.
    """
    width1, height1 = rect1.right - rect1.left, rect1.bottom - rect1.top
    width2, height2 = rect2.right - rect2.left, rect2.bottom - rect2.top

    # Calculate the area of intersection
    intersection_area = min(width1, width2) * min(height1, height2)

    # Calculate the areas of the two rectangles
    area_rect1 = width1 * height1
    area_rect2 = width2 * height2

    # Calculate the Union area
    union_area = area_rect1 + area_rect2 - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area
    # print("iou: ", iou)

    return iou

def vis_bbox(frame, curr_box, old_box):
    print(curr_box, old_box)
    image = frame.copy()
    x, y, x2, y2 = curr_box.left, curr_box.top, curr_box.right, curr_box.bottom    
    cv2.rectangle(image, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
    x, y, x2, y2 = old_box.left, old_box.top, old_box.right, old_box.bottom    
    cv2.rectangle(image, (int(x), int(y)), (int(x2), int(y2)), (255, 0, 0), 2)
    plt.imshow(image)
    plt.show()
    
def get_contact_frame(annots, frames, frames_idxs, hand_threshold, obj_threshold, hand_side):
    # print("frames.shape: ", len(frames))
    for i in range(5, len(annots)):
        # print(f"---- frame number {i} ----")
        # get the ith hand and object
        curr_hand, curr_obj, curr_obj_bbox = get_hand_object(annots[i], hand_threshold, obj_threshold, hand_side)
        # print("curr_hand, curr_obj, curr_obj_bbox: ", curr_hand, curr_obj, curr_obj_bbox)
        # means there was no hand-obj correspondence for this hand so no need to check
        if curr_hand is None:
            continue
        # FIRST CHECK: check previous frames: x-8 to x-3. I take this range because the model predicts that there is an interaction between
        # the hand and object too soon. And so if I don't do this, the contact frame chosen would be too early of a frame!
        total_frames_of_contact = 0
        for j in range(max(i-5, 0), i):
            old_hand, old_obj, old_obj_bbox = get_hand_object(annots[j], hand_threshold, obj_threshold, hand_side)
            # means old_hand was in contact. Now need to check if it was the same object that was in contact
            if old_hand is not None:
                iou = calculate_iou_sizes(curr_obj_bbox, old_obj_bbox)
                # if i == 75:
                #     print("iou: ", iou)
                # vis_bbox(frames[i], curr_obj_bbox, old_obj_bbox)
                if iou > 0.3:
                    total_frames_of_contact += 1
        # print("total_frames_of_contact for previous frames: ", total_frames_of_contact)
        if total_frames_of_contact >= 2:
            continue

        # SECOND CHECK: check successive frames - should be in contact with the same object for the next 5 frames.
        frames_len = len(frames)
        total_frames_of_contact = 0
        for j in range(min(i+1, frames_len), min(i+6, frames_len)):
            new_hand, new_obj, new_obj_bbox = get_hand_object(annots[j], hand_threshold, obj_threshold, hand_side)
            # -------- remove later -------
            # print("Displaying image")
            # x = renderer.render_detections(PIL.Image.fromarray(frames[j]), annots[j])
            # plt.imshow(np.array(x))
            # plt.show()
            # -----------------------------
            if new_hand is not None:
                iou = calculate_iou_sizes(curr_obj_bbox, new_obj_bbox)
                if iou > 0.5:
                    total_frames_of_contact += 1
        # print("total_frames_of_contact for the next frames: ", total_frames_of_contact)
        if total_frames_of_contact < 4:
            continue
        # If passes the two checks, this is the contact frame!
        contact_frame_idx = i + 5
        # plt.imshow(frames[contact_frame_idx])
        # plt.show()
        return contact_frame_idx, frames_idxs[contact_frame_idx]
    print("-------------Did not find any new contact frame---------")
    return None

def check_contact(annots, frames, frame_idx, hand_side):
    print("inside check contact")
    counter = 0
    renderer = DetectionRenderer(hand_threshold=0.5, object_threshold=0.5)
    if frame_idx < 2:
        return False
    else:
        for i in range(frame_idx-2, frame_idx+3):
            x = renderer.render_detections(PIL.Image.fromarray(frames[i]), annots[i])
            plt.imshow(np.array(x))
            plt.show()
            hand = None
            for h in annots[i].hands:
                # print("------", h.side.name, hand_side)
                if h.side.name == hand_side:
                    hand = h
                    break
            if hand is None:
                continue
            print("hand.state.name: ", hand.state.name)
            if hand.state.name != 'NO_CONTACT':
                counter += 1
    print("------number of hand contacts: ", counter)
    if counter > 2:
        return True
    else:
        return False


def get_start_frame(contact_frame_idx_right, contact_frame_idx_left, frames_idxs, annots):
    if contact_frame_idx_left < contact_frame_idx_right:
        start_frame_idx_left = 0
        start_frame_left = frames_idxs[0]
        start_frame_idx_right = max(0, contact_frame_idx_right - 10)
        # for i in range(contact_frame_idx_right-10):
        #     is_contact = check_contact(annots, frame_idx=contact_frame_idx_right-10-i, hand_side='RIGHT')
        #     if not is_contact:
        #         start_frame_idx_right = i
        #         break
        
        start_frame_right = frames_idxs[start_frame_idx_right]
    else:
        start_frame_idx_right = 0
        start_frame_right = frames_idxs[0]
        start_frame_idx_left = max(0, contact_frame_idx_left - 10)
        # for i in range(contact_frame_idx_left-10):
        #     is_contact = check_contact(annots, frame_idx=contact_frame_idx_left-10-i, hand_side='LEFT')
        #     if not is_contact:
        #         start_frame_idx_left = i
        #         break
        
        start_frame_left = frames_idxs[start_frame_idx_left]
    
    return start_frame_idx_right, start_frame_right, start_frame_idx_left, start_frame_left

def prepare_hand_mask(frames, annots):
    save_path = '/home/arpit/test_projects/ProPainter/inputs/object_removal/'
    # frame = cv2.imread('/home/arpit/test_projects/ProPainter/inputs/object_removal/bmx-trees_mask/00001.png')
    # print("frame: ", frame, frame[150:160][200:220].shape)
    # plt.imshow(frame)
    # plt.show()
    counter = 0
    for idx, annot in enumerate(annots):
        print("idx: ", idx)
        hand = None
        for h in annot.hands:
            if h.side.name == 'LEFT':
                hand = h
        if hand is not None:
            hand_bbox = hand.bbox.coords
            print("hand_bbox: ", hand_bbox)
            x1, y1, x2, y2 = int(hand_bbox[0][0]), int(hand_bbox[0][1]), int(hand_bbox[1][0]), int(hand_bbox[1][1])
            hand_mask_img = np.zeros_like(frames[idx])
            print("hand_mask_img.shape: ", hand_mask_img.shape)
            hand_mask_img = cv2.rectangle(hand_mask_img, (x1, y1), (x2, y2), (255, 255, 255), -1)
            # fig, ax = plt.subplots(1,2)
            # ax[0].imshow(frames[idx])
            # ax[1].imshow(hand_mask_img)
            # plt.show()
            img_path = save_path + f"ek2/{counter:05d}.jpg"
            mask_path = save_path + f"ek2_mask/{counter:05d}.png"
            print("img_path: ", img_path)
            cv2.imwrite(img_path, frames[idx])
            cv2.imwrite(mask_path, hand_mask_img)
            counter += 1

            


def fetch_data(frames_path, video_detections, frames_idxs, wandb_table, wandb_log=False, save_video=False, hand_threshold=0.5, obj_threshold=0.5):
    frames = []
    annots = []

    miss_hand = 0
    counter = 0
    # if want to save video
    save_path = f'arpit_output/hoa_videos'
    height, width = 256, 456
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_file = save_path + f'/{video_detections[0].video_id}_{frames_idxs[0]}_hoa.mp4'
    output_video = cv2.VideoWriter(output_video_file, fourcc, 1, (width, height))

    for frame_idx in frames_idxs:
        # print(f"----------------- frame {frame_idx},{counter} -------------------")
        frame = load_img(frames_path, frame_idx)
        annot = load_ho_annot(video_detections, frame_idx)
        # print("annot: ", annot)
        hands = [hand for hand in annot.hands if hand.score >= hand_threshold]
        # # --- remove later------
        # renderer = DetectionRenderer(hand_threshold=0.5, object_threshold=0.5)
        # x = renderer.render_detections(PIL.Image.fromarray(frame), annot)
        # plt.imshow(np.array(x))
        # plt.show()
        # # ------
        if len(hands) == 0:
            miss_hand += 1
        frames.append(frame)
        # plt.imshow(frame)
        # plt.show()
        annots.append(annot)
        counter += 1
    if miss_hand == len(frames_idxs[:-1]):
        print("Since no hands detected in this video, skipping it!")
        return None

    # prepare_hand_mask(frames, annots)

    renderer = DetectionRenderer(hand_threshold=0.5, object_threshold=0.5)
    video_id, start_frame = video_detections[0].video_id, frames_idxs[0]
    contact_frame_idx_right, contact_frame_right = None, None
    contact_frame_idx_left, contact_frame_left = None, None
    
    # get contact frame for the right hand
    retval = get_contact_frame(annots, frames, frames_idxs, hand_threshold, obj_threshold, hand_side='RIGHT')
    # print("retval: ", retval)
    if retval is not None:
        # confirm if the contact frame is correct
        contact_frame_idx_right, contact_frame_right = retval
        # print("First Predicted Right Hand Contact Frame")
        # plt.imshow(frames[contact_frame_idx_right])
        # plt.show()
        # retval = get_contact_frame(annots[contact_frame_idx_right:], frames[contact_frame_idx_right:], frames_idxs[contact_frame_idx_right:], hand_threshold, obj_threshold, hand_side='RIGHT')
        # if retval is not None:
        #     print("retval is not None")
        #     new_contact_frame_idx_right, new_contact_frame_right = retval
        #     contact_frame_idx_right = contact_frame_idx_right + new_contact_frame_idx_right
        #     # for kk in range(contact_frame_idx_right-10, contact_frame_idx_right-5):
        #     #     x = renderer.render_detections(PIL.Image.fromarray(frames[kk]), annots[kk])
        #     #     plt.imshow(np.array(x))
        #     #     plt.show()           
        #     # print("Second Predicted Right Hand Contact Frame")
        #     # plt.imshow(frames[contact_frame_idx_right])
        #     # plt.show()            
        # else:
        #     # print("Original contact frame was correct!!!")
        #     pass
        print("contact_frame_idx_right: ", contact_frame_idx_right)
        # plt.imshow(frames[contact_frame_idx_right])
        # plt.show()
    else:
        print(f"get_contact_frame failed for right hand for video: {video_id}_{start_frame}.mp4")
        return None
        

    # get contact frame for the left hand
    retval = get_contact_frame(annots, frames, frames_idxs, hand_threshold, obj_threshold, hand_side='LEFT')
    if retval is not None:
        # confirm if the contact frame is correct
        contact_frame_idx_left, contact_frame_left = retval
        # print("First Predicted Left Hand Contact Frame")
        # plt.imshow(frames[contact_frame_idx_left])
        # plt.show()
        # retval = get_contact_frame(annots[contact_frame_idx_left:], frames[contact_frame_idx_left:], frames_idxs[contact_frame_idx_left:], hand_threshold, obj_threshold, hand_side='LEFT')
        # if retval is not None:
        #     # print("retval is not None")
        #     new_contact_frame_idx_left, new_contact_frame_left = retval
        #     contact_frame_idx_left = contact_frame_idx_left + new_contact_frame_idx_left
        #     # print("Second Predicted Left Hand Contact Frame, ", new_contact_frame_idx_left)            
        #     # plt.imshow(frames[contact_frame_idx_left])
        #     # plt.show()
        # f = contact_frame_idx_left-5
        # x = renderer.render_detections(PIL.Image.fromarray(frames[f]), annots[f])
        # plt.imshow(np.array(x))
        # plt.show()
        # for kk in range(f-5, f):
        #     x = renderer.render_detections(PIL.Image.fromarray(frames[kk]), annots[kk])
        #     plt.imshow(np.array(x))
        #     plt.show()
        # print("----------")   
        # for kk in range(f+1, f+6):
        #     x = renderer.render_detections(PIL.Image.fromarray(frames[kk]), annots[kk])
        #     plt.imshow(np.array(x))
        #     plt.show()            
        # else:
        #     # print("Original contact frame was correct!!!")
        #     pass
        print("contact_frame_idx_left: ", contact_frame_idx_left)
        # plt.imshow(frames[contact_frame_idx_left])
        # plt.show()
    else:
        print(f"get_contact_frame failed for left hand for video: {video_id}_{start_frame}.mp4")
        return None

    start_frame_idx_right, start_frame_right, start_frame_idx_left, start_frame_left  = get_start_frame(contact_frame_idx_right,
                                                                                                        contact_frame_idx_left,
                                                                                                        frames_idxs, annots)
    print('start_frame_idx_right, start_frame_idx_left: ', start_frame_idx_right, start_frame_idx_left)
    # start_frame_is_contact_right = check_contact(annots, start_frame_idx_right, 'RIGHT')
    start_frame_is_contact_left = check_contact(annots, frames, start_frame_idx_left, 'LEFT')
    if start_frame_is_contact_left:
        return None
   
    # fig, ax = plt.subplots(2,2)
    # ax[0, 0].imshow(frames[start_frame_idx_right])
    # ax[0, 1].imshow(frames[contact_frame_idx_right])
    # ax[1, 0].imshow(frames[start_frame_idx_left])
    # ax[1, 1].imshow(frames[contact_frame_idx_left])
    # plt.show()       
    
    # temp_frames = [frames[start_frame_idx_left], frames[contact_frame_idx_left]]
    # temp_annots = [annots[start_frame_idx_left], annots[contact_frame_idx_left]]
    # prepare_hand_mask(temp_frames, temp_annots)
    
    if save_video:
        # right hand video
        output_video_file_right = create_output_video(frames_path,
                            frames_idxs,
                            start_frame_img=frames[start_frame_idx_right],
                            contact_frame_img=frames[contact_frame_idx_right],
                            save_path='arpit_output/contact_frames',
                            video_id=video_id,
                            start_frame=frames_idxs[0],
                            wandb_table=wandb_table,
                            hand_side='right')  
        # left hand video
        output_video_file_left = create_output_video(frames_path,
                            frames_idxs,
                            start_frame_img=frames[start_frame_idx_left],
                            contact_frame_img=frames[contact_frame_idx_left],
                            save_path='arpit_output/contact_frames',
                            video_id=video_id,
                            start_frame=frames_idxs[0],
                            wandb_table=wandb_table,
                            hand_side='left')
    if wandb_log:
        wandb_table.add_data(f'{video_id}_{start_frame}.mp4\n [start_frame, contact_frame, video]',
                            wandb.Video(output_video_file_left, fps=30, format="mp4"),
                            wandb.Video(output_video_file_right, fps=30, format="mp4"))  
        
    frames_idxs_left = frames_idxs[start_frame_idx_left:contact_frame_idx_left+1]
    frames_left = frames[start_frame_idx_left:contact_frame_idx_left+1]
    annots_left = annots[start_frame_idx_left:contact_frame_idx_left+1]
    frames_idxs_right = frames_idxs[start_frame_idx_right:contact_frame_idx_right+1]
    frames_right = frames[start_frame_idx_right:contact_frame_idx_right+1]
    annots_right = annots[start_frame_idx_right:contact_frame_idx_right+1]
    return frames_idxs_left, frames_left, annots_left, frames_idxs_right, frames_right, annots_right
    # print("FINAL: ", frames_idxs, len(frames), len(annots), list(set(hand_sides)))
    # return frames_idxs, frames, annots, list(set(hand_sides)) # remove redundant hand sides


def save_video_info(save_path, video_index, frames_idxs, homography_stack, contacts,
               hand_trajs, obj_trajs, affordance_info):
    import pickle
    video_info = {"frame_indices": frames_idxs,
                  "homography": homography_stack,
                  "contact": contacts}
    video_info.update({"hand_trajs": hand_trajs})
    video_info.update({"obj_trajs": obj_trajs})
    video_info.update({"affordance": affordance_info})
    with open(os.path.join(save_path, "label_{}.pkl".format(video_index)), 'wb') as f:
        pickle.dump(video_info, f)


def load_video_info(save_path, video_index):
    import pickle
    with open(os.path.join(save_path, "label_{}.pkl".format(video_index)), 'rb') as f:
        video_info = pickle.load(f)
    return video_info
