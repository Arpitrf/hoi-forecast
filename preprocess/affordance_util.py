import cv2
import numpy as np
from preprocess.dataset_util import bbox_inter
import matplotlib.pyplot as plt

def skin_extract2(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    final_image = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)
    non_zero_mask = np.all(final_image != [0, 0, 0], axis=-1)
    # print("non_zero_mask: ", non_zero_mask)
    final_mask = np.zeros_like(non_zero_mask, dtype=np.uint8)
    # print("-----------final_mask: ", final_mask)
    final_mask[non_zero_mask] = 255
    # print("Other skin extaction method")
    # plt.imshow(final_mask)
    # plt.show()

    # Return the Skin image
    return final_mask

def skin_extract(image):
    def color_segmentation():
        lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
        upper_HSV_values = np.array([25, 255, 255], dtype="uint8")
        lower_YCbCr_values = np.array((0, 138, 67), dtype="uint8")
        upper_YCbCr_values = np.array((255, 173, 133), dtype="uint8")
        mask_YCbCr = cv2.inRange(YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
        mask_HSV = cv2.inRange(HSV_image, lower_HSV_values, upper_HSV_values)
        binary_mask_image = cv2.add(mask_HSV, mask_YCbCr)
        return binary_mask_image

    HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    YCbCr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    binary_mask_image = color_segmentation()
    image_foreground = cv2.erode(binary_mask_image, None, iterations=3)
    dilated_binary_image = cv2.dilate(binary_mask_image, None, iterations=3)
    ret, image_background = cv2.threshold(dilated_binary_image, 1, 128, cv2.THRESH_BINARY)

    image_marker = cv2.add(image_foreground, image_background)
    image_marker32 = np.int32(image_marker)
    cv2.watershed(image, image_marker32)
    m = cv2.convertScaleAbs(image_marker32)
    ret, image_mask = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((20, 20), np.uint8)
    image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_CLOSE, kernel)
    return image_mask


def farthest_sampling(pcd, n_samples, init_pcd=None):
    def compute_distance(a, b):
        return np.linalg.norm(a - b, ord=2, axis=2)

    n_pts, dim = pcd.shape[0], pcd.shape[1]
    selected_pts_expanded = np.zeros(shape=(n_samples, 1, dim))
    remaining_pts = np.copy(pcd)

    if init_pcd is None:
        if n_pts > 1:
            start_idx = np.random.randint(low=0, high=n_pts - 1)
        else:
            start_idx = 0
        selected_pts_expanded[0] = remaining_pts[start_idx]
        n_selected_pts = 1
    else:
        num_points = min(init_pcd.shape[0], n_samples)
        selected_pts_expanded[:num_points] = init_pcd[:num_points, None, :]
        n_selected_pts = num_points

    for _ in range(1, n_samples):
        if n_selected_pts < n_samples:
            dist_pts_to_selected = compute_distance(remaining_pts, selected_pts_expanded[:n_selected_pts]).T
            dist_pts_to_selected_min = np.min(dist_pts_to_selected, axis=1, keepdims=True)
            res_selected_idx = np.argmax(dist_pts_to_selected_min)
            selected_pts_expanded[n_selected_pts] = remaining_pts[res_selected_idx]
            n_selected_pts += 1

    selected_pts = np.squeeze(selected_pts_expanded, axis=1)
    return selected_pts


def compute_heatmap(points, image_size, k_ratio=3.0):
    points = np.asarray(points)
    heatmap = np.zeros((image_size[0], image_size[1]), dtype=np.float32)
    n_points = points.shape[0]
    for i in range(n_points):
        x = points[i, 0]
        y = points[i, 1]
        col = int(x)
        row = int(y)
        try:
            heatmap[col, row] += 1.0
        except:
            col = min(max(col, 0), image_size[0] - 1)
            row = min(max(row, 0), image_size[1] - 1)
            heatmap[col, row] += 1.0
    k_size = int(np.sqrt(image_size[0] * image_size[1]) / k_ratio)
    if k_size % 2 == 0:
        k_size += 1
    heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = heatmap.transpose()
    return heatmap


def select_points_bbox(bbox, points, tolerance=2):
    x1, y1, x2, y2 = bbox
    ind_x = np.logical_and(points[:, 0] > x1-tolerance, points[:, 0] < x2+tolerance)
    ind_y = np.logical_and(points[:, 1] > y1-tolerance, points[:, 1] < y2+tolerance)
    ind = np.logical_and(ind_x, ind_y)
    indices = np.where(ind == True)[0]
    return points[indices]


def find_contour_points(mask):
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        c = c.squeeze(axis=1)
        return c
    else:
        return None


def get_points_homo(select_points, homography, active_obj_traj, obj_bboxs_traj, start_frame, contact_frame):
    # active_obj_traj: active obj traj in last observation frame
    # obj_bboxs_traj: active obj bbox traj in last observation frame
    vis0 = vis_points(start_frame.copy(), select_points)
    contact_frame_img = vis_points(contact_frame.copy(), select_points)
    select_points_homo = np.concatenate((select_points, np.ones((select_points.shape[0], 1), dtype=np.float32)), axis=1)
    select_points_homo = np.dot(select_points_homo, homography.T)
    select_points_homo = select_points_homo[:, :2] / select_points_homo[:, None, 2]
    
    # print("1 len(select_points_homo): ", len(select_points_homo))
    vis1 = vis_points(start_frame.copy(), select_points_homo)

    obj_point_last_observe = np.array(active_obj_traj[0]) # start frame
    obj_point_future_start = np.array(active_obj_traj[-1]) # contact frame
    point1 = obj_point_last_observe.astype(np.int)
    vis2 = cv2.circle(start_frame.copy(), (point1[0], point1[1]), radius=2, color=(0, 0, 255), thickness=-1)
    point2 = obj_point_future_start.astype(np.int)
    vis2 = cv2.circle(vis2, (point2[0], point2[1]), radius=2, color=(255, 0, 0), thickness=-1)

    future2last_trans = obj_point_last_observe - obj_point_future_start
    select_points_homo = select_points_homo + future2last_trans

    # print("2 len(select_points_homo): ", len(select_points_homo))
    vis3 = vis_points(start_frame.copy(), select_points_homo)

    fill_indices = [idx for idx, points in enumerate(obj_bboxs_traj) if points is not None]
    contour_last_observe = obj_bboxs_traj[fill_indices[0]]
    contour_future_homo = obj_bboxs_traj[fill_indices[-1]] + future2last_trans

    # # remove later
    # contour_last_observe[0][0] -= 10
    # contour_last_observe[0][1] -= 10
    # contour_last_observe[1][0] += 10
    # contour_last_observe[1][1] -= 10
    # contour_last_observe[2][0] -= 10
    # contour_last_observe[2][1] += 10
    # contour_last_observe[3][0] += 10
    # contour_last_observe[3][1] += 10

    # print("---------------contor_last_observe: ", contour_last_observe, contour_future_homo)
    # Fixing the bug of HOI: The polygon formed wasn't a rectangle before because the ordering of the corners of the rectangle was - 0, 1, 3, 2 
    # instead of 0, 1, 2, 3
    contour_last_observe[[2,3]] = contour_last_observe[[3,2]]
    contour_future_homo[[2,3]] = contour_future_homo[[3,2]]
    
    contour_last_observe = contour_last_observe[:, None, :].astype(np.int)
    contour_future_homo = contour_future_homo[:, None, :].astype(np.int)
    filtered_points = []
    for point in select_points_homo:
        # if cv2.pointPolygonTest(contour_future_homo, (point[0], point[1]), False) >= 0:
        print("point: ", point, cv2.pointPolygonTest(contour_last_observe, (point[0], point[1]), False))
        if cv2.pointPolygonTest(contour_last_observe, (point[0], point[1]), False) >= 0 \
                or cv2.pointPolygonTest(contour_future_homo, (point[0], point[1]), False) >= 0:   
            filtered_points.append(point)
    
    # print("3 len(filtered_points): ", len(filtered_points))
    vis4 = vis_bbox(start_frame.copy(), [np.squeeze(contour_last_observe), np.squeeze(contour_future_homo)], coord=True)
    vis4 = vis_points(vis4.copy(), filtered_points)
    vis5 = cv2.polylines(start_frame.copy(), [contour_last_observe], 
                      True, (255, 0, 0), 2)
    vis5 = vis_points(vis5.copy(), select_points_homo)


    # fig, ax = plt.subplots(2,3)
    # ax[0,0].imshow(contact_frame_img)
    # ax[0,1].imshow(vis5)
    # ax[0,2].imshow(vis1)
    # ax[1,0].imshow(vis2)
    # ax[1,1].imshow(vis3)
    # ax[1,2].imshow(vis4)
    # plt.show()
    
    filtered_points = np.array(filtered_points)
    return filtered_points

def vis_masks(frame, mask, mask_name):
    color_mask = np.stack((mask,) * 3, axis=-1)
    print("Showing: ", mask_name)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(frame)
    ax[1].imshow(color_mask)
    plt.show()
    frame_plus_mask = (0.3 * frame + 0.7 * color_mask).astype(np.uint8)
    plt.imshow(frame_plus_mask)
    plt.show()
    # return frame_plus_mask

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

def vis_points(frame, pts, extra_pt=None):
    frame_pts = frame.copy()
    for pt in pts:
        point = pt.astype(np.int)
        frame_pts = cv2.circle(frame_pts, (point[0], point[1]), radius=1, color=(255, 0, 255), thickness=-1)
    if extra_pt is not None:
        print("extra_point: ", extra_pt)
        extra_point = np.asarray(extra_pt).astype(np.int)
        frame_pts = cv2.circle(frame_pts, (extra_point[0], extra_point[1]), radius=1, color=(0, 0, 255), thickness=-1)
    return frame_pts
    # plt.imshow(frame_pts)
    # plt.show()

def compute_affordance(frame, active_hand, active_obj, num_points=5, num_sampling=20, start_frame=None):
    # skin_mask = skin_extract(frame)
    # print("skin_mask shape1111111: ", type(skin_mask), type(skin_mask[0][0]), skin_mask.shape)
    skin_mask = skin_extract2(frame)
    # print("skin_mask222 shape22222222: ", type(skin_mask), type(skin_mask[0][0]), skin_mask.shape)
    hand_bbox = np.array(active_hand.bbox.coords_int).reshape(-1)
    obj_bbox = np.array(active_obj.bbox.coords_int).reshape(-1)
    # print("hand_bbox, obj_bbox: ", hand_bbox.shape, obj_bbox.shape)
    # vis_bbox(frame, np.array([hand_bbox, obj_bbox]))
    obj_center = active_obj.bbox.center 
    xA, yA, xB, yB, iou = bbox_inter(hand_bbox, obj_bbox)
    print("showing hand and object bbox intersection with IOU: ", iou)
    vis_bbox(frame, np.array([[xA, yA, xB, yB]]))
    if not iou > 0:
        return None
    x1, y1, x2, y2 = hand_bbox
    hand_mask = np.zeros_like(skin_mask, dtype=np.uint8)
    hand_mask[y1:y2, x1:x2] = 255
    org_hand_mask = hand_mask
    hand_mask = cv2.bitwise_and(skin_mask, hand_mask)
    # print("hand_mask: ", hand_mask)
    # vis_masks(frame, skin_mask, "skin_mask")
    # vis_masks(frame, hand_mask, "hand_mask")

    select_points, init_points = None, None
    contact_points = find_contour_points(hand_mask)
    all_hand_contour_points = contact_points
    # print("contact_points: ", contact_points.shape)
    # vis = vis_points(frame, contact_points)
    # plt.imshow(vis)
    # plt.show()

    if contact_points is not None and contact_points.shape[0] > 0:
        contact_points = select_points_bbox((xA, yA, xB, yB), contact_points)
        contact_points_in_bbox_inter = contact_points
        # print("contact_points after bbox constraint: ", contact_points.shape)
        # vis = vis_points(frame, contact_points)
        # plt.imshow(vis)
        # plt.show()
        if contact_points.shape[0] >= num_points:
            if contact_points.shape[0] > num_sampling:
                contact_points = farthest_sampling(contact_points, n_samples=num_sampling)
            contact_points_after_farthest_sampling = contact_points
            # vis_points(frame, contact_points)
            distance = np.linalg.norm(contact_points - obj_center, ord=2, axis=1)
            indices = np.argsort(distance)[:num_points]
            select_points = contact_points[indices]
        elif contact_points.shape[0] > 0:
            print("no enough boundary points detected, sampling points in interaction region")
            init_points = contact_points
        else:
            print("no boundary points detected, use farthest point sampling")
    else:
        print("no boundary points detected, use farthest point sampling")
    if select_points is None:
        print("-----Select points was None------")
        ho_mask = np.zeros_like(skin_mask, dtype=np.uint8)
        ho_mask[yA:yB, xA:xB] = 255
        ho_mask = cv2.bitwise_and(skin_mask, ho_mask)
        # vis_masks(frame, ho_mask, "ho_mask")
        points = np.array(np.where(ho_mask[yA:yB, xA:xB] > 0)).T
        # print("points.shape: ", points.shape)
        # vis_points(frame, points)
        if points.shape[0] == 0:
            xx, yy = np.meshgrid(np.arange(xB - xA), np.arange(yB - yA))
            xx += xA
            yy += yA
            points = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T
        else:
            # print("points11: ", points[:5])
            points = points[:, [1, 0]]
            # print("points22: ", points[:5])
            points[:, 0] += xA
            points[:, 1] += yA
            # vis_points(frame, points)
        if not points.shape[0] > 0:
            return None
        contact_points = farthest_sampling(points, n_samples=min(num_sampling, points.shape[0]), init_pcd=init_points)
        contact_points_after_farthest_sampling = contact_points
        # print("contact_points: ", contact_points.shape)
        # vis_points(frame, contact_points, obj_center)
        distance = np.linalg.norm(contact_points - obj_center, ord=2, axis=1)
        indices = np.argsort(distance)[:num_points]
        select_points = contact_points[indices]
    
    # ---------------- Vis ---------------------
    # org_hand_mask_color = np.stack((org_hand_mask,) * 3, axis=-1)
    # skin_mask_color = np.stack((skin_mask,) * 3, axis=-1)
    # final_hand_mask_color = np.stack((hand_mask,) * 3, axis=-1)
    # org_hand_mask_img = (0.3 * frame + 0.7 * org_hand_mask_color).astype(np.uint8)
    # skin_mask_img = (0.3 * frame + 0.7 * skin_mask_color).astype(np.uint8)
    # final_hand_mask_img = (0.3 * frame + 0.7 * final_hand_mask_color).astype(np.uint8)
    # vis1 = vis_points(frame, all_hand_contour_points)
    # vis2 = vis_points(frame, contact_points_in_bbox_inter)
    # vis3 = vis_points(frame, contact_points_after_farthest_sampling)
    # vis4 = vis_points(frame, select_points)
    # fig, ax = plt.subplots(2,4)
    # ax[0,0].imshow(org_hand_mask_img)
    # ax[0,1].imshow(skin_mask_img)
    # ax[0,2].imshow(final_hand_mask_img)
    # ax[1,0].imshow(vis1)
    # ax[1,1].imshow(vis2)
    # ax[1,2].imshow(vis3)
    # ax[1,3].imshow(vis4)    
    # plt.show()
    # ------------------------------------------
    
    return select_points


def compute_obj_affordance(frame, annot, active_obj, active_obj_idx, homography,
                           active_obj_traj, obj_bboxs_traj,
                           num_points=10, num_sampling=20,
                           hand_threshold=0.5, obj_threshold=0.5, start_frame=None, hand_side=None):
    affordance_info = {}
    hand_object_idx_correspondences = annot.get_hand_object_interactions(object_threshold=obj_threshold,
                                                                         hand_threshold=hand_threshold)
    # print("hoa correspondences: ", [(annot.hands[k].side, v) for k,v in hand_object_idx_correspondences.items()])
    select_points = None 
    for hand_idx, object_idx in hand_object_idx_correspondences.items():
        print("**********", annot.hands[hand_idx].side.name)
        if hand_side != annot.hands[hand_idx].side.name:
            continue
        if object_idx == active_obj_idx:
            active_hand = annot.hands[hand_idx]
            affordance_info[active_hand.side.name] = np.array(active_hand.bbox.coords_int).reshape(-1)
            # Q. What are these cmap_points and select_points?
            cmap_points = compute_affordance(frame, active_hand, active_obj, num_points=num_points, num_sampling=num_sampling, start_frame=start_frame)
            print("cmap_points: ", cmap_points)
            if select_points is None and (cmap_points is not None and cmap_points.shape[0] > 0):
                select_points = cmap_points
            elif select_points is not None and (cmap_points is not None and cmap_points.shape[0] > 0):
                select_points = np.concatenate((select_points, cmap_points), axis=0)
    if select_points is None:
        print("affordance contact points filtered out")
        return None
    print("len(select_points): ", len(select_points))
    select_points_homo = get_points_homo(select_points, homography, active_obj_traj, obj_bboxs_traj, start_frame.copy(), frame.copy())
    print("len(select_points_homo): ", len(select_points_homo))
    round_select_points_homo = np.round(select_points_homo).astype(int)
    # print("select_points_homo: ", round_select_points_homo)
    
    # # ------------ vis cmap points ------------
    # frame_vis = frame.copy()
    # for cmap_point in cmap_points.astype(int):
    #     frame_vis = cv2.circle(frame_vis, (cmap_point[0], cmap_point[1]), radius=2, color=(255, 0, 255),
    #                     thickness=-1)
    # frame_vis2 = start_frame.copy()
    # for select_point_homo in round_select_points_homo:
    #     frame_vis2 = cv2.circle(frame_vis2, (select_point_homo[0], select_point_homo[1]), radius=2, color=(255, 0, 255),
    #                     thickness=-1)
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(frame_vis)
    # ax[1].imshow(frame_vis2)
    # plt.show()
    # # -----------------------------------------

    if len(select_points_homo) == 0:
        print("affordance contact points filtered out")
        return None
    else:
        affordance_info["select_points"] = select_points
        affordance_info["select_points_homo"] = select_points_homo

        obj_bbox = np.array(active_obj.bbox.coords_int).reshape(-1)
        affordance_info["obj_bbox"] = obj_bbox
        return affordance_info