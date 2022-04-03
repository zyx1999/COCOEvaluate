import os
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class COCOEvaluate:
    def __init__(self, gt_anno_path):
        self._cat_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        self.coco = COCO(gt_anno_path)
        self.coco_image_ids = self.coco.getImgIds()

    def get_eval_image_ids(self, images_dir):
        """
        获取待测图像的image_id, 返回由image_id组成的list

        images_dir: str
        
        """
        print("get [image_id] for evaluation ...")
        image_ids = []
        for dir_path, dir_name, files in os.walk(images_dir):
            for file in files:
                if file.endswith('.jpg'):
                    image_ids.append(file)
        return image_ids


    def evaluate(self, mdc_json_path, image_ids):
        """
        mdc_json_path: str,
            MDC输出数据转换成COCO Result格式的json文件所在路径
        
        images_ids: [str, ..., str],
            由待测试图片的image_id组成的数组 
        """
        whole_eval_ids = {
            self.coco.loadImgs(img_id)[0]["file_name"]: img_id
            for img_id in self.coco_image_ids
        }
        cls_ids = list(range(1, 80 + 1))
        eval_ids = [whole_eval_ids[image_id] for image_id in image_ids]
        cat_ids  = [self._classes[cls_id] for cls_id in cls_ids]

        coco_dets = self.coco.loadRes(mdc_json_path)
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0]

    def to_coco_results_format(self, dst_json_path, src_dir_path, image_ids, score_threshold=0):
        """
        将mdc的输出数据转换成COCO Results格式
        
        dst_json_path: str,
            数据格式转换完成生成的json文件的保存路径
        
        src_dir_path: str,
            MDC的输出数据的存放路径

        images_ids: [str, ..., str],
            由待测试图片的image_id组成的数组 

        score_threshold: float, default=0,
            用于过滤mdc的输出数据中score=0.0的bbox
        """
        print("[CONVERT] convert ascend_out to coco result format ...")
        whole_eval_ids = {
            self.coco.loadImgs(img_id)[0]["file_name"]: img_id
            for img_id in self.coco_image_ids
        }
        results = []
        for image_id in image_ids:
            # print('image_id: {}'.format(image_id))
            coco_id = whole_eval_ids[image_id]
            dets_path = src_dir_path+image_id[:-4]+'_out.bin'
            detections = np.fromfile(dets_path, dtype=np.float16).reshape(1024, 8).tolist()
            for each_det in detections:
                score = each_det[4]
                if score > score_threshold:
                    x1 = each_det[0]
                    y1 = each_det[1]
                    w = each_det[2] - x1
                    h = each_det[3] - y1
                    bbox = list([x1, y1, w, h])
                    bbox = [float("{:.2f}".format(x)) for x in bbox]
                    category_id = self._classes[int(each_det[5])+1]
                    coco_result_format = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }
                    results.append(coco_result_format)
        with open(dst_json_path, "w") as f:
            json.dump(results, f)
            print("[CONVERT] finished, json文件保存至: {}".format(dst_json_path))

    def rename_ascend_out(self, data_dir):
        print("[RENAME] rename *_out.bin in ascend_out/ ...")
        for dir_path, dir_name, files in os.walk(data_dir):
            for file in files:
                # print("mv {} {}".format(data_dir+file, data_dir+file[:27]+'_out.bin'))
                os.rename(data_dir+file,data_dir+file[:27]+'_out.bin')
        print("[RENAME] finished")

    def compare_ap(self, ap_torch, ap_om):
        print("AP | torch | om ")
        print("   | {} | {} ".format(float("{:.3f}".format(ap_torch)), float("{:.3f}".format(ap_om))))





