from evaluate import COCOEvaluate

if __name__=="__main__":
    # COCO 官方标注(train2014)
    gt_anno_path = './annotations/annotations_2014/instances_train2014.json'

    # 数据经过 "pytorch模型" 输出的json文件
    dets_json_path = './results/train2014/torch_results_100_imgs.json'

    # 数据经过 "om模型" 输出的json文件
    mdc_json_path = './results/train2014/mdc_result_100_imgs.json'

    # 用于测试的图像和特征的存放路径
    images_dir='./images_with_features/'

    # om模型输出数据存放路径
    ascend_out_dir_path = './ascend_out/'

    # 1. 读入 COCO官方标注
    cocoeval = COCOEvaluate(gt_anno_path)

    # 2. 获取待测图像的image_id
    eval_image_ids = cocoeval.get_eval_image_ids(images_dir)
    
    # 3. 将ascend_out/中的算子输出文件更名
    cocoeval.rename_ascend_out(data_dir=ascend_out_dir_path)
    
    # 4. 将ascend_out/中的算子输出文件转换为COCO格式
    cocoeval.to_coco_results_format(mdc_json_path, ascend_out_dir_path, eval_image_ids)
    
    # 5. 计算 "pytorch模型" mAP
    ap_torch = cocoeval.evaluate(dets_json_path, eval_image_ids)
    
    # 6. 计算 "om模型" mAP
    ap_om = cocoeval.evaluate(mdc_json_path, eval_image_ids)

    # 7. 输出mAP
    cocoeval.compare_ap(ap_torch=ap_torch, ap_om=ap_om)