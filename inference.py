import os
import json
import torch
import torch.nn as nn
import numpy as np
from pycocotools.coco import COCO
from external.nms import soft_nms

def _gather_feat(feat, ind, mask=None):                                   # eg: feat = [2,128*128,2]  ind = [2,100]
    dim  = feat.size(2)                                                   # eg: dim = 2
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)         # eg: ind = [2,100,2]
    feat = feat.gather(1, ind)                                            # eg: feat = [2,100,2]
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _tranpose_and_gather_feat(feat, ind):                                                # eg: feat = [2,128*128,2]  ind = [2,100] 
    feat = feat.permute(0, 2, 3, 1).contiguous()   # [0,1,2,3] -> [0,2,3,1]  列变换         eg: [2,2,128,128] -> [2,128,128,2]
    feat = feat.view(feat.size(0), -1, feat.size(3)) # [0,2 * 3,1]  不是2*3，是第二列和第三列合并   eg: [2,128*128,2]
    feat = _gather_feat(feat, ind)                                                       # eg: feat = [2,100,2]
    return feat

def _topk(scores, K=20):    # because K = 100, all = [2,100]  top_K选出K个
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def crop_image(new_size, center, inp_size):
    cty, ctx            = center
    height, width       = inp_size
    im_height, im_width = new_size
    
    x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
    y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)

    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty

    cropped_cty, cropped_ctx = height // 2, width // 2

    border = torch.FloatTensor([
       cropped_cty - top,
       cropped_cty + bottom,
       cropped_ctx - left,
       cropped_ctx + right
    ])

    return border

def calrb(size):
    new_height, new_width = size
    
    new_center = torch.Tensor([new_height // 2, new_width // 2])

    inp_height = new_height | 127
    inp_width  = new_width  | 127

    ratios  = torch.zeros((1, 2), dtype=torch.float32)
    borders = torch.zeros((1, 4), dtype=torch.float32)

    out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
    height_ratio = out_height / inp_height
    width_ratio  = out_width  / inp_width

    border = crop_image(size, new_center, [inp_height, inp_width])
    
    borders[0] = border
    ratios[0]  = torch.FloatTensor([height_ratio, width_ratio])
    return ratios, borders

def _rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs    /= ratios[:, 1][:, None, None]
    ys    /= ratios[:, 0][:, None, None]
    xs    -= borders[:, 2][:, None, None]
    ys    -= borders[:, 0][:, None, None]
    xs = np.clip(xs, 0, sizes[:, 1][:, None, None])
    ys = np.clip(ys, 0, sizes[:, 0][:, None, None])

def _to_float(x):
        return float("{:.2f}".format(x))

def convert_to_coco(all_bboxes):
    _cat_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
            82, 84, 85, 86, 87, 88, 89, 90
        ]
    _classes = {
        ind + 1: cat_id for ind, cat_id in enumerate(_cat_ids)
    }
    # coco    = COCO('./annotations/instances_testdev2017.json')
    coco    = COCO('./annotations/annotations_2014/instances_train2014.json')

    # cat_ids = coco.getCatIds()
    coco_image_ids = coco.getImgIds()
    whole_eval_ids = {
        coco.loadImgs(img_id)[0]["file_name"]: img_id
        for img_id in coco_image_ids
    }
    detections = []
    for image_id in all_bboxes:
        # print('image_id', image_id)
        coco_id = whole_eval_ids[image_id]
        # print('coco_id', coco_id)
        for cls_ind in all_bboxes[image_id]:
            category_id = _classes[cls_ind]
            # print('==category_id', category_id)
            for bbox in all_bboxes[image_id][cls_ind]:
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                score = bbox[4]
                bbox  = list(map(_to_float, bbox[0:4]))

                detection = {
                    "image_id": coco_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": float("{:.2f}".format(score))
                }

                detections.append(detection)
    return detections

def _decode(det_out_dir, image_id,
    tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, sizes, merge_bbox=False, num_dets=1000,
    K=100, kernel=3, ae_threshold=0.5, categories=80, nms_threshold=0.5, max_per_image=100
):
    top_bboxes = {}
    batch, cat, height, width = tl_heat.size()  # heat = [2,80,heigth,width] (heigth,width is random, and this is [128,192]) 
    # tl_tag = [2,1,heigth,width]   tl_regr = [2,2,heigth,width]
    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)
    
    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K) # tl_scores ...and tl_xs = [2,100]  top_K选出100个
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    
    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)    # [2,100,100]
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)

    if tl_regr is not None and br_regr is not None:
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)   # [2,100,2]
        tl_regr = tl_regr.view(batch, K, 1, 2)                  # [2,100,1,2]
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)   # [2,100,2]
        br_regr = br_regr.view(batch, 1, K, 2)                  # [2,1,100,2]

        tl_xs = tl_xs + tl_regr[..., 0]                         
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]                         # br_regr[..., 0] = [2,1,100]  相加时维度扩展，维度为1的复制为100
        br_ys = br_ys + br_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
    tl_tag = tl_tag.view(batch, K, 1)
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag = br_tag.view(batch, 1, K)
    dists  = torch.abs(tl_tag - br_tag)

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores    = (tl_scores + br_scores) / 2

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    width_inds  = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)

    scores[cls_inds]    = -1
    scores[dist_inds]   = -1
    scores[width_inds]  = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)  # [2,1000,1]

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds) # [2,1000,4]

    clses  = tl_clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()

    dets = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)  #detections = [2,1000,8]  bboxes = [2,1000,4]
    # return detections
    dets   = dets.view(1, -1, 8)

    # cal borders, ratios
    new_height = sizes[0][0]
    new_width = sizes[0][1]
    
    ratios, borders = calrb([int(new_height), int(new_width)])

    _rescale_dets(dets, ratios, borders, sizes)

    out_dets = dets.numpy().astype(np.float16)
    # print("detectsion.shape: {}".format(out_dets.shape))
    os.mkdir('../detections_100_imgs/'+det_out_dir+'/')
    save_dets_path = '../detections_100_imgs/'+det_out_dir+'/'+image_id[0:-4]+'.bin'
    print("save detections to: ", save_dets_path)
    out_dets.tofile(save_dets_path)

    detections = []
    detections.append(dets.numpy())
    detections = np.concatenate(detections, axis=1)    

    classes    = detections[..., -1]
    classes    = classes[0]    # shape = (1,2000)  -> (2000)
    detections = detections[0]  # shape = (2000,8)

    # reject detections with negative scores
    keep_inds  = (detections[:, 4] > -1)
    detections = detections[keep_inds]
    classes    = classes[keep_inds]
    
    # print("detections")
    # print(detections)

    # dets = dets.numpy()
    top_bboxes[image_id] = {}
    for j in range(categories):
        # 取第j类的框的下标
        keep_inds = (classes == j)
        # 取第j类的框
        top_bboxes[image_id][j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)  # exclude classes
        soft_nms(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=0)
        top_bboxes[image_id][j + 1] = top_bboxes[image_id][j + 1][:, 0:5]

    scores = np.hstack([
        top_bboxes[image_id][j][:, -1] 
        for j in range(1, categories + 1)
    ])
    if len(scores) > max_per_image:
        kth    = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories + 1):
            keep_inds = (top_bboxes[image_id][j][:, -1] >= thresh)
            top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]
    
    return top_bboxes

if __name__ == "__main__":
    root_dir = '../train_100_imgs/'
    top_bboxes = {}
    case_num = 99
    for i in range(1, case_num+1):
        if i<10:
            _dir = '00'+str(i)
        elif i < 100:
            _dir = '0'+str(i)
        elif i >=100:
            _dir = str(i)
        dir = root_dir+_dir+'/'
        print(dir)
    # for case_name in case_list:
        # dir = './traindataset/test_batch1/COCO_train2014_000000'+case_name+'/'
        # prefix = '2014_000000'+case_name
        tl_heat_path = dir+'tlh.bin'
        br_heat_path = dir+'brh.bin'
        tl_tag_path  = dir+'tlt.bin'
        br_tag_path  = dir+'brt.bin'
        tl_regr_path = dir+'tlr.bin'
        br_regr_path = dir+'brr.bin'

        # image_id = 'COCO_train2014_000000'+case_name+'.jpg'
        for dir_path, dir_name, files in os.walk(dir):
            for file in files:
                if file.endswith('.jpg'):
                    image_id = file

        tl_heat = np.fromfile(tl_heat_path, dtype=np.float32).reshape((1,80,128,128))
        br_heat = np.fromfile(br_heat_path, dtype=np.float32).reshape((1,80,128,128))
        tl_tag  = np.fromfile(tl_tag_path,  dtype=np.float32).reshape((1,1,128,128))
        br_tag  = np.fromfile(br_tag_path,  dtype=np.float32).reshape((1,1,128,128))
        tl_regr = np.fromfile(tl_regr_path, dtype=np.float32).reshape((1,2,128,128))
        br_regr = np.fromfile(br_regr_path, dtype=np.float32).reshape((1,2,128,128))
        aimg = _decode(_dir, image_id, torch.tensor(tl_heat), torch.tensor(br_heat), torch.tensor(tl_tag), torch.tensor(br_tag), 
            torch.tensor(tl_regr), torch.tensor(br_regr), torch.from_numpy(np.array([[500, 500]])))
        top_bboxes.update(aimg)

    print(top_bboxes)
    result_json = os.path.join('./results/train2014/torch_results_100_imgs.json')
    detections  = convert_to_coco(top_bboxes)
    with open(result_json, "w") as f:
        json.dump(detections, f)
