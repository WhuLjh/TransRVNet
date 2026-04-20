import os
from skimage import measure
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
import warnings
# warnings.filterwarnings("ignore")

def segment_watershed(binary_image):
    # Compute the distance transform
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255,
                               0)  # Lower threshold to detect small objects
    sure_fg = np.uint8(sure_fg)

    # Refine foreground markers
    kernel = np.ones((3, 3), np.uint8)
    sure_fg = cv2.dilate(sure_fg, kernel, iterations=2)  # More dilation to preserve small objects
    unknown = cv2.subtract(binary_image, sure_fg)

    # Ensure small objects are recognized
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    binary_image_colored = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    cv2.watershed(binary_image_colored, markers)

    return markers

def extract_skeleton(segmented_image):
    unique_labels = np.unique(segmented_image)
    skeletons = {}
    for label_id in unique_labels:
        if label_id > 1:
            region_mask = (segmented_image == label_id).astype(np.uint8)
            skeleton = skeletonize(region_mask)
            skeletons[label_id] = skeleton
    return skeletons

def sample_skeleton(skeleton):
    coords = np.argwhere(skeleton > 0)
    if len(coords) < 1:
        return coords

    # Find endpoints
    endpoints = np.array([[x, y] for x, y in coords if np.sum(skeleton[max(0, x - 1):x + 2, max(0, y - 1):y + 2]) == 2])

    # Ensure certain number of skeleton points are sampled
    num_samples = max(1, len(coords) // 20)  # max(1, len(coords) // 10)  2
    sampled_indices = np.linspace(0, len(coords) - 1, num=num_samples, dtype=int)
    sampled_points = coords[sampled_indices]

    # Ensure folding points are included (junctions)
    fold_points = np.array(
        [p for p in coords if np.sum(skeleton[max(0, p[0] - 1):p[0] + 2, max(0, p[1] - 1):p[1] + 2]) > 3])

    # Combine all sampled points
    # all_samples = np.vstack([arr for arr in [sampled_points, endpoints, fold_points, center] if arr.size > 0])
    all_samples = np.vstack([arr for arr in [sampled_points, endpoints, fold_points] if arr.size > 0])

    return np.unique(all_samples, axis=0)

def hyperpixel_segmentation(rgb_image, skeleton_points,pre_mask, save_name=None, num_segments=100):
    segments = slic(rgb_image, n_segments=num_segments, compactness=10, sigma=1)
    hyperpixel_array = np.zeros(rgb_image.shape[:2], dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(50, 50))
    bmask = cv2.dilate(pre_mask, kernel,3).astype(np.int64)
    bmask[bmask==0] = -1
    bmask[bmask>0]=1
    segments = segments * bmask
    segments[segments<0] = -1

    skeleton_set = set(map(tuple, skeleton_points))
    segment_labels = np.unique(segments)

    for segment_id in segment_labels:
        if segment_id != -1:
            segment_mask = (segments == segment_id)
            if any(tuple(pt) in skeleton_set for pt in np.argwhere(segment_mask)):
                hyperpixel_array[segment_mask] = 1

    if save_name != None:
        out = mark_boundaries(rgb_image, segments,color=(1, 1, 1))
        out = out * 211  #
        out = out.astype(np.uint8)
        plt.figure(figsize=(8, 8))
        plt.imshow(out)
        #### Pre_mask contours
        contours_pre = measure.find_contours(pre_mask, 0.5)
        for n, contour in enumerate(contours_pre):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color='black')
        #### Hyper_mask contours
        contours = measure.find_contours(hyperpixel_array, 0.5)
        for n, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=3,color='red')
        #### Sample points
        plt.plot([v[1] for v in skeleton_points],[v[0] for v in skeleton_points],'o',markersize=2,color='gold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_name)
        plt.close()

    return hyperpixel_array

def SkeLine_connected_SLIC(images, output_dec1,name,epoch,segments,savedir,save_vis):
    corrected_pseudo_label = np.zeros((output_dec1.shape[0], output_dec1.shape[1], output_dec1.shape[2]), dtype=np.uint8)
    for b in range(images.shape[0]):
        img = images[b,:,:,:].detach().numpy()
        pred = output_dec1[b,:,:].astype(np.uint8)
        if len(np.unique(pred))==1 and np.unique(pred)[0] == 0:
            corrected_pseudo_label[b, :, :] = pred
        else:
            _, binary_image = cv2.threshold(pred, 0, 1, cv2.THRESH_BINARY)
            segmented = segment_watershed(binary_image)
            skeletons = extract_skeleton(segmented)

            if len(skeletons) > 0:
                save_name = None
                masks = pred
                ####-------------SkeLine-connected SLIC----------------
                skeleton_points = np.vstack([np.argwhere(skel) for skel in skeletons.values()])
                hyperpixel_result = hyperpixel_segmentation(img, skeleton_points, masks, save_name, segments)

                ####-------------------------Save_vis:start---------------------------------
                if save_vis == True:
                    savedir3 = str(savedir) +'\\process\\process_vis\\'+ str(epoch) + '\\'
                    if not os.path.exists(savedir3):
                        os.makedirs(savedir3)
                    if epoch in [0, 4, 9]:
                        save_name = savedir3 + name[b]
                        plt.figure(figsize=(8, 8))
                        plt.imshow(img)
                        #### Pre_mask contours
                        contours_pre = measure.find_contours(pred, 0.5)
                        for n, contour in enumerate(contours_pre):
                            plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color='black')
                        #### Hyper_mask contours
                        contours = measure.find_contours(masks, 0.5)
                        for n, contour in enumerate(contours):
                            plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color='red')
                        #### Sample lines
                        skeleton_points = np.vstack([np.argwhere(skel) for skel in skeletons.values()])
                        plt.plot([v[1] for v in skeleton_points], [v[0] for v in skeleton_points], 'o', markersize=5,
                                 color='blue')
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(save_name)
                        plt.close()
                        ####-------------------------Save_vis:end---------------------------------

                masks = pred + hyperpixel_result
                masks[masks > 0] = 1
            else:
                masks = pred

            corrected_pseudo_label[b,:,:] = masks

        out = corrected_pseudo_label[b,:,:].copy()
        out[out == 1] = 255
        savedir1 = savedir+'\\process\\pre_pl\\' + str(epoch)+'\\'
        savedir2 = savedir+'\\process\\pre_pl_color\\' + str(epoch)+'\\'

        if not os.path.exists(savedir1):
            os.makedirs(savedir1)
        if not os.path.exists(savedir2):
            os.makedirs(savedir2)
        cv2.imwrite(savedir1+name[b], corrected_pseudo_label[b,:,:])
        cv2.imwrite(savedir2+name[b], out)

    return corrected_pseudo_label

def SkePoint_prompted_BufSAM(images, output_dec1, predictor,name,epoch,buffer,savedir,save_vis):
    corrected_pseudo_label = np.zeros((output_dec1.shape[0], output_dec1.shape[1], output_dec1.shape[2]), dtype=np.uint8)
    for b in range(images.shape[0]):
        img = images[b,:,:,:].detach().numpy()
        pred = output_dec1[b,:,:].astype(np.uint8)
        if len(np.unique(pred))==1 and np.unique(pred)[0] == 0:
            corrected_pseudo_label[b, :, :] = pred
        else:
            predictor.set_image(img)
            ############ Positive points
            _, binary_image = cv2.threshold(pred, 0, 1, cv2.THRESH_BINARY)
            segmented = segment_watershed(binary_image)
            skeletons = extract_skeleton(segmented)

            if len(skeletons) > 0:
                #####-------------SkePoint-prompted BufSAM----------------
                skeleton_samples = {label_id: sample_skeleton(skel) for label_id, skel in skeletons.items()}
                values = list(skeleton_samples.values())
                indexs = np.array([j for i in values for j in i])

                labels = []
                new_ = []
                for i in indexs:
                    new_.append([i[1], i[0]])
                    labels.append(1)

                indexs_ = np.array(new_)
                labels_ = np.array(labels)
                sampled_indices = np.linspace(0, len(indexs_) - 1, num=int(len(indexs_)), dtype=int)
                indexs_ = indexs_[sampled_indices]
                labels_ = labels_[sampled_indices]

                # Additional points
                input_point = indexs_
                input_label = labels_

                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    # mask_input=mask_input[None, :, :],
                    multimask_output=True,
                )
                area = np.sum(np.sum(masks, 1), 1)
                masks = masks[area.argmin(), :, :] ## Select the small one

                ##### buffers
                if buffer[0] != 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, buffer)
                    pred_dilated = cv2.dilate(pred, kernel, 3)
                    masks = masks * pred_dilated

                ####-------------------------Save_vis:start---------------------------------
                if save_vis == True:
                    savedir3 = str(savedir) +'\\process\\process_vis\\'+ str(epoch) + '\\'
                    if not os.path.exists(savedir3):
                        os.makedirs(savedir3)
                    if epoch in [0, 4, 9]:
                        save_name = savedir3 + name[b]
                        plt.figure(figsize=(8, 8))
                        plt.imshow(img)
                        #### Pre_mask contours
                        contours_pre = measure.find_contours(pred, 0.5)
                        for n, contour in enumerate(contours_pre):
                            plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color='black')
                        #### Hyper_mask contours
                        contours = measure.find_contours(masks, 0.5)
                        for n, contour in enumerate(contours):
                            plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color='red')
                        #### Sample points
                        skeleton_points = np.vstack([skel for skel in skeleton_samples.values()])
                        plt.plot([v[1] for v in skeleton_points], [v[0] for v in skeleton_points], 'o',
                                 markersize=5,
                                 color='gold')
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(save_name)
                        plt.close()
                        ####-------------------------Save_vis:end---------------------------------

                masks = pred + masks
                masks[masks > 0] = 1
            else:
                masks = pred

            corrected_pseudo_label[b,:,:] = masks

        out = corrected_pseudo_label[b,:,:].copy()
        out[out == 1] = 255
        savedir1 = savedir + '\\process\\pre_pl\\' + str(epoch) + '\\'
        savedir2 = savedir + '\\process\\pre_pl_color\\' + str(epoch) + '\\'

        if not os.path.exists(savedir1):
            os.makedirs(savedir1)
        if not os.path.exists(savedir2):
            os.makedirs(savedir2)
        cv2.imwrite(savedir1 + name[b], corrected_pseudo_label[b, :, :])
        cv2.imwrite(savedir2 + name[b], out)

    return corrected_pseudo_label
