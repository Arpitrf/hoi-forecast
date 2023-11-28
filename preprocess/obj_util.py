import numpy as np
from preprocess.dataset_util import bbox_inter, HandState, compute_iou, \
    valid_traj, get_valid_traj, points_in_bbox
from preprocess.traj_util import get_homo_point, get_homo_bbox_point
from hoa.visualisation import DetectionRenderer
import PIL
import matplotlib.pyplot as plt
import cv2

def vis_bbox(frame, boxes, coord=False):
    image = frame.copy()
    if coord:
        for i, box in enumerate(boxes):
            color = (i*50, 0 , 0)
            # Determine the top-left and bottom-right coordinates of the bounding box
            # print("-------", box[0][0], box[1][0], box[2][0], box[3][0])
            x_min = min(box[0][0], box[1][0], box[2][0], box[3][0])
            y_min = min(box[0][1], box[1][1], box[2][1], box[3][1])
            x_max = max(box[0][0], box[1][0], box[2][0], box[3][0])
            y_max = max(box[0][1], box[1][1], box[2][1], box[3][1])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)  # (0, 255, 0) is the color, 2 is the thickness

    else:
        for box in boxes:
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            x, y, x2, y2 = box[0], box[1], box[2], box[3]
            cv2.rectangle(image, (int(x), int(y)), (int(x2), int(y2)), color, 2)
    # plt.imshow(image)
    # plt.show()
    return image

# TODO: Both hands can also be active I guess?
def find_active_side(annots, hand_sides, hand_threshold=0.1, obj_threshold=0.1):
    if len(hand_sides) == 1:
        return hand_sides[0]
    else:
        hand_counter = {"LEFT": 0, "RIGHT": 0}
        for annot in annots:
            hands = [hand for hand in annot.hands if hand.score >= hand_threshold]
            objs = [obj for obj in annot.objects if obj.score >= obj_threshold]
            if len(hands) > 0 and len(objs) > 0:
                # obtain the hand and the object that probably the hand is interacting with
                hand_object_idx_correspondences = annot.get_hand_object_interactions(object_threshold=obj_threshold,
                                                                                     hand_threshold=hand_threshold)
                for hand_idx, object_idx in hand_object_idx_correspondences.items():
                    hand_bbox = np.array(annot.hands[hand_idx].bbox.coords_int).reshape(-1)
                    obj_bbox = np.array(annot.objects[object_idx].bbox.coords_int).reshape(-1)
                    xA, yA, xB, yB, iou = bbox_inter(hand_bbox, obj_bbox)
                    # if the hand and the corresponding object has a bbox overlap, the correspondign frame adds to the "activeness" of the hand   
                    if iou > 0:
                        hand_side = annot.hands[hand_idx].side.name
                        # print("annot.hands[hand_idx].state.value: ", hand_side, annot.hands[hand_idx].state.value)
                        if annot.hands[hand_idx].state.value == HandState.PORTABLE_OBJECT.value:
                            hand_counter[hand_side] += 1
                        elif annot.hands[hand_idx].state.value == HandState.STATIONARY_OBJECT.value:
                            hand_counter[hand_side] += 0.5
                # print("----------")
        # print("hand_counter: ", hand_counter)
        if hand_counter["LEFT"] == hand_counter["RIGHT"]:
            return "RIGHT"
        else:
            return max(hand_counter, key=hand_counter.get)


def compute_contact(annots, hand_side, contact_state, hand_threshold=0.1):
    contacts = []
    for annot in annots:
        hands = [hand for hand in annot.hands if hand.score >= hand_threshold
                 and hand.side.name == hand_side and hand.state.value == contact_state]
        if len(hands) > 0:
            contacts.append(1)
        else:
            contacts.append(0)
    contacts = np.array(contacts)
    padding_contacts = np.pad(contacts, [1, 1], 'edge')
    contacts = np.convolve(padding_contacts, [1, 1, 1], 'same')
    contacts = contacts[1:-1] / 3
    contacts = contacts > 0.5
    indices = np.diff(contacts) != 0
    if indices.sum() == 0:
        return contacts
    else:
        split = np.where(indices)[0] + 1
        contacts_idx = split[-1]
        contacts[:contacts_idx] = False
        return contacts


def find_active_obj_side(annot, hand_side, return_hand=False, return_idx=False, hand_threshold=0.1, obj_threshold=0.1):
    objs = [obj for obj in annot.objects if obj.score >= obj_threshold]
    if len(objs) == 0:
        return None
    else:
        hand_object_idx_correspondences = annot.get_hand_object_interactions(object_threshold=obj_threshold,
                                                                             hand_threshold=hand_threshold)
        for hand_idx, object_idx in hand_object_idx_correspondences.items():
            if annot.hands[hand_idx].side.name == hand_side:
                if return_hand and return_idx:
                    return annot.objects[object_idx], object_idx, annot.hands[hand_idx], hand_idx
                elif return_hand:
                    return annot.objects[object_idx], annot.hands[hand_idx]
                elif return_idx:
                    return annot.objects[object_idx], object_idx
                else:
                    return annot.objects[object_idx]
        return None


def find_active_obj_iou(objs, bbox):
    max_iou = 0
    active_obj = None
    for obj in objs:
        iou = compute_iou(obj.bbox.coords, bbox)
        if iou > max_iou:
            max_iou = iou
            active_obj = obj
    return active_obj, max_iou


def traj_compute(annots, hand_sides, homography_stack, hand_threshold=0.1, obj_threshold=0.1, frames=None):
    annot = annots[-1]
    obj_traj = []
    obj_centers = []
    obj_bboxs =[]
    obj_bboxs_traj = []
    active_hand_side = find_active_side(annots, hand_sides, hand_threshold=hand_threshold,
                                        obj_threshold=obj_threshold)
    # print("active_hand_side: ", active_hand_side)
    retval = find_active_obj_side(annot,
                                hand_side=active_hand_side,
                                return_hand=True, return_idx=True,
                                hand_threshold=hand_threshold,
                                obj_threshold=obj_threshold)
    if retval is None:
        return None
    active_obj, active_object_idx, active_hand, active_hand_idx = retval
    # print("active_obj: ", active_obj, active_object_idx, active_hand, active_hand_idx)
    contact_state = active_hand.state.value
    contacts = compute_contact(annots, active_hand_side, contact_state,
                               hand_threshold=hand_threshold)
    # print("contacts: ", contacts)
    obj_center = active_obj.bbox.center
    obj_centers.append(obj_center)
    obj_point = get_homo_point(obj_center, homography_stack[-1])
    obj_bbox = active_obj.bbox.coords
    obj_traj.append(obj_point)
    obj_bboxs.append(obj_bbox)

    obj_points2d = get_homo_bbox_point(obj_bbox, homography_stack[-1])
    obj_bboxs_traj.append(obj_points2d)

    for idx in np.arange(len(annots)-2, -1, -1):
        annot = annots[idx]
        objs = [obj for obj in annot.objects if obj.score >= obj_threshold]
        contact = contacts[idx]
        # if not contact:
        #     print("in hereeeee0: ", idx)
        #     obj_centers.append(None)
        #     obj_traj.append(None)
        #     obj_bboxs_traj.append(None)
        # else:
        # remove later
        # vis_boxes = []
        # if idx <= 42:
        #     for b in objs:
        #         print("b.bbox: ", b.bbox.coords)
        #         box = [b.bbox.coords[0][0], b.bbox.coords[0][1], b.bbox.coords[1][0], b.bbox.coords[1][1]]
        #         vis_boxes.append(box)
        #     img = vis_bbox(frames[idx], vis_boxes)
        #     plt.imshow(img)
        #     plt.show()

        if len(objs) >= 2:
            target_obj, max_iou = find_active_obj_iou(objs, obj_bboxs[-1])
            # if target_obj is None:
            #     target_obj = find_active_obj_side(annot, hand_side=active_hand_side,
            #                                         hand_threshold=hand_threshold,
            #                                         obj_threshold=obj_threshold)
            if target_obj is None:
                # print("in hereeeee1: ", idx)
                obj_centers.append(None)
                obj_traj.append(None)
                obj_bboxs_traj.append(None)
            else:
                obj_center = target_obj.bbox.center
                obj_centers.append(obj_center)
                obj_point = get_homo_point(obj_center, homography_stack[idx])
                obj_bbox = target_obj.bbox.coords
                obj_traj.append(obj_point)
                obj_bboxs.append(obj_bbox)

                obj_points2d = get_homo_bbox_point(obj_bbox, homography_stack[idx])
                obj_bboxs_traj.append(obj_points2d)

        elif len(objs) > 0:
            target_obj, max_iou = find_active_obj_iou(objs, obj_bboxs[-1])
            # if target_obj is None:
            #     target_obj = find_active_obj_side(annot, hand_side=active_hand_side,
            #                                     hand_threshold=hand_threshold,
            #                                     obj_threshold=obj_threshold)
            if target_obj is None:
                # print("in hereeeee2: ", idx)
                obj_centers.append(None)
                obj_traj.append(None)
                obj_bboxs_traj.append(None)
            else:
                obj_center = target_obj.bbox.center
                obj_centers.append(obj_center)
                obj_point = get_homo_point(obj_center, homography_stack[idx])
                obj_bbox = target_obj.bbox.coords
                obj_traj.append(obj_point)
                obj_bboxs.append(obj_bbox)

                obj_points2d = get_homo_bbox_point(obj_bbox, homography_stack[idx])
                obj_bboxs_traj.append(obj_points2d)
        else:
            # print("in hereeeee3: ", idx)
            obj_centers.append(None)
            obj_traj.append(None)
            obj_bboxs_traj.append(None)
        # # remove later
        # if obj_traj[-1] is not None:
        #     print("obj_traj[-1] is not None: ", idx)
        #     vis_img = frames[idx]
        #     vis_img = cv2.circle(vis_img, (int(obj_traj[-1][0]), int(obj_traj[-1][1])), radius=2, color=(255, 0, 0), thickness=-1)
        #     vis_img = cv2.circle(vis_img, (int(obj_centers[-1][0]), int(obj_centers[-1][1])), radius=2, color=(0, 0, 255), thickness=-1)
        #     x, y, x2, y2 = obj_bboxs[-1][0][0], obj_bboxs[-1][0][1], obj_bboxs[-1][1][0], obj_bboxs[-1][1][1]
        #     vis_img = cv2.rectangle(vis_img, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
        #     plt.imshow(vis_img)
        #     plt.show()
        # else:
        #     print("obj_traj[-1] is None: ", idx)
    obj_bboxs.reverse()
    obj_traj.reverse()
    obj_centers.reverse()
    obj_bboxs_traj.reverse()
    
    # remove later
    # print("obj_traj: ", obj_traj[0][0], type(obj_traj[0][0]))
    # print("-------")
    # obj_centers_np = np.array(obj_centers).astype(np.float32)
    # obj_traj = obj_centers_np
    # print("obj_centers: ", obj_centers_np[0], type(obj_centers_np[0][0]))
    return obj_traj, obj_centers, obj_bboxs, contacts, active_obj, active_object_idx, obj_bboxs_traj


def traj_filter(obj_traj, obj_centers, obj_bbox, contacts, homography_stack, contact_ratio=0.4):
    assert len(obj_traj) == len(obj_centers), "traj length and center length not equal"
    assert len(obj_centers) == len(homography_stack), "center length and homography length not equal"
    homo_last2first = homography_stack[-1]
    homo_first2last = np.linalg.inv(homo_last2first)
    obj_points = []
    obj_inside, obj_detect = [], []
    for idx, obj_center in enumerate(obj_centers):
        if obj_center is not None:
            homo_current2first = homography_stack[idx]
            homo_current2last = homo_current2first.dot(homo_first2last)
            obj_point = get_homo_point(obj_center, homo_current2last)
            obj_points.append(obj_point)
            obj_inside.append(points_in_bbox(obj_point, obj_bbox))
            obj_detect.append(True)
        else:
            obj_detect.append(False)
    obj_inside = np.array(obj_inside)
    obj_detect = np.array(obj_detect)
    contacts = np.bitwise_and(obj_detect, contacts)
    if np.sum(obj_inside) == len(obj_inside) and np.sum(contacts) / len(contacts) < contact_ratio:
        obj_traj = np.tile(obj_traj[-1], (len(obj_traj), 1))
    return obj_traj, contacts


def traj_completion(traj, imgW=456, imgH=256):
    fill_indices = [idx for idx, point in enumerate(traj) if point is not None]
    full_traj = traj.copy()
    if len(fill_indices) == 1:
        point = traj[fill_indices[0]]
        full_traj = np.array([point] * len(traj), dtype=np.float32)
    else:
        contact_time = fill_indices[0]
        if contact_time > 0:
            full_traj[:contact_time] = [traj[contact_time]] * contact_time
        for previous_idx, current_idx in zip(fill_indices[:-1], fill_indices[1:]):
            start_point, end_point = traj[previous_idx], traj[current_idx]
            time_expand = current_idx - previous_idx
            for idx in range(previous_idx+1, current_idx):
                full_traj[idx] = (idx-previous_idx) / time_expand * end_point + (current_idx-idx) / time_expand * start_point
    full_traj = np.array(full_traj, dtype=np.float32)
    full_traj = get_valid_traj(full_traj, imgW=imgW, imgH=imgH)
    return full_traj, fill_indices


def compute_obj_traj(frames, annots, hand_sides, homography_stack, hand_threshold=0.1, obj_threshold=0.1,
                     contact_ratio=0.4):
    # To confirm if hand, object and hand-object detections are fine
    # # ----- remove later ------
    # for i in range(len(frames)):
    #     print("idx: ", i)
    #     frame = frames[i]
    #     annot = annots[i]
    #     renderer = DetectionRenderer(hand_threshold=0.5, object_threshold=0.5)
    #     x = renderer.render_detections(PIL.Image.fromarray(frame), annot)
    #     plt.imshow(np.array(x))
    #     plt.show()
    # # -------------------------
    imgH, imgW = frames[0].shape[:2]
    retval = traj_compute(annots, hand_sides, homography_stack,
                            hand_threshold=hand_threshold, obj_threshold=obj_threshold, frames=frames)
    if retval is None:
        return None    
    obj_traj, obj_centers, obj_bboxs, contacts, active_obj, active_object_idx, obj_bboxs_traj = retval
    obj_traj, contacts = traj_filter(obj_traj, obj_centers, obj_bboxs[-1], contacts, homography_stack,
                                     contact_ratio=contact_ratio)
    obj_traj = valid_traj(obj_traj, imgW=imgW, imgH=imgH)
    if len(obj_traj) == 0:
        print("object traj filtered out")
        return None
    else:
        complete_traj, fill_indices = traj_completion(obj_traj, imgW=imgW, imgH=imgH)
        obj_trajs = {"traj": complete_traj, "fill_indices": fill_indices, "centers": obj_centers}
        return contacts, obj_trajs, active_obj, active_object_idx, obj_bboxs_traj