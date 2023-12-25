import torch
import cv2
import numpy as np
from data import FaceDetection, FaceDetection_vol1, detection_collate, preproc, cfg_mnet, cfg_re50
from models.retinaface import RetinaFace
from collections import OrderedDict
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import jaccard
import torch.nn.functional as F

from tqdm.auto import tqdm
import os
import datetime
import csv


def read_image(image_path, device, target_size, rgb_mean=(104, 117, 123)):

    img_raw = cv2.imread(image_path)  # Read in BGR
    img = np.float32(img_raw)
    # resize image
    # print(img.shape)
    im_height, im_width, _ = img.shape
    resize = min(target_size / im_height, target_size / im_width)

    if resize < 1.0:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

    # Pad the image to achieve a fixed size of 640x640 without distortion
    pad_height = target_size - img.shape[0]
    pad_width = target_size - img.shape[1]

    if pad_height > 0 or pad_width > 0:
        # img = cv2.copyMakeBorder(img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img = cv2.copyMakeBorder(img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=rgb_mean)
    # print(img.shape)
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= rgb_mean
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    return img, img_raw, scale, scale1, resize

def get_envelope(precisions: np.ndarray) -> np.array:
    """Compute the envelope of the precision curve.
    Args:
      precisions:
    Returns: enveloped precision
    """
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    return precisions

def get_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Calculate area under precision/recall curve.
    Args:
      recalls:
      precisions:
    Returns:
    """
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    precisions = get_envelope(precisions)

    # to calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap

def decode_and_NMS(loc, conf, landms, prior_data, cfg, scale, scale1, resize, confidence_threshold, nms_threshold):
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])  # [16800, 4]
    # boxes = loc.data.squeeze(0)  # [16800, 4]  for IOU Loss
    boxes = boxes * scale / resize
    # boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]  # (16800,)
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])  # [16800, 10]
    landms = landms * scale1 / resize
    # landms = landms * scale1
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]  # n
    boxes = boxes[inds]  # (n,4)
    landms = landms[inds]  # (n,10)
    scores = scores[inds]  # (n)

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    # order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]  # (n', 5)
    landms = landms[keep]  # (n', 10)

    dets = np.concatenate((dets, landms), axis=1)  # (n', 15)
    return dets

def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def get_centroid_distance(image, box, centroid_pos=[0.5, 0.3]):
    if len(box.shape) == 1:
        box = np.expand_dims(box, axis=0)
    h, w = image.shape[:2]
    centroid = np.zeros((len(box), 2))
    centroid[:, 0] = (box[:, 0] + box[:, 2]) / 2
    centroid[:, 1] = (box[:, 1] + box[:, 3]) / 2
    distance = np.linalg.norm(centroid - np.array([w * centroid_pos[0], h * centroid_pos[1]]), axis=1)
    return distance

def load_model(cfg, mode, device, model_weight_path='./weights/mobilenet0.25_Final_Repo.pth'):
    net = RetinaFace(cfg=cfg, phase=mode)
    print('Loading resume network for 0.25 ...')
    state_dict = torch.load(model_weight_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net = net.to(device)
    if mode != 'train':
        net.eval()

    return net

def infer(net, image_path, img_dim, device, cfg, confidence_threshold = None, nms_threshold =None):
    img, img_raw, scale, scale1, resize = read_image(image_path, device, target_size=img_dim)
    # forward pass
    loc, conf, landms = net(img)
    # print(loc.shape, conf.shape, landms.shape)
    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(device)
    prior_data = priors.data
    if confidence_threshold is None:
        confidence_threshold = cfg['infer_confidence_threshold']
    if nms_threshold is None:
        nms_threshold = cfg['infer_nms_threshold']
    dets = decode_and_NMS(loc, conf, landms, prior_data, cfg, scale, scale1, resize, confidence_threshold, nms_threshold)
    return dets, img_raw

# save image
def save_image(dets, cfg, img_raw, idx):
    for b in dets:
        if b[4] < cfg['infer_confidence_threshold']:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        print('b: ', b)
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

    if not os.path.exists("./results/"):
        os.makedirs("./results/")
    name = "./results/" + str(idx) + ".jpg"
    cv2.imwrite(name, img_raw)


# output labels for each image (4 for bbox, 1 for confidence score, 5 for landms)
def output_label(pred, img_path, res_file_path):
    try:
        with open(res_file_path, "a") as fd:
            fd.write('# ' + img_path + '\n')
            fd.write(' '.join(str(num) for num in pred[0]) + '\n')
    except Exception as e:
        print("error: ", e)


def eval_main(mode,  data_folder_path, label_files, cfg, net = None, net_ckpt_path = None , log_file_path= None, epoch = None, save_fp_fn=False, fp_output_file= "fp.csv", fn_output_file = "fn.csv", closest_face_only=False, limit=None, confidence_threshold = None, nms_threshold =None):
    write_to_log(log_file_path, f"[{mode}], dataset: {label_files}")
    # load trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if net is None:
        print("No model being passed, reading from checkpoints:", net_ckpt_path)
        net = load_model(cfg, mode, device, model_weight_path=net_ckpt_path)
    else:
        print("Evaluating passed model for " + mode)
        net = net.eval()

    # load data
    dataset = FaceDetection(label_folder_path=data_folder_path, label_files=label_files, preproc=None)
    dataset_imgs = dataset.imgs_path

    # constants
    if confidence_threshold is None:
        confidence_threshold = cfg['infer_confidence_threshold']
    if nms_threshold is None:
        nms_threshold = cfg['infer_nms_threshold']  # 0.4
    iou_thresh = cfg['infer_iou_thresh']  # 0.5
    img_dim = cfg['infer_image_size']
    print(f"img_dim:{str(img_dim)}")
    print(f"iou_thresh:{iou_thresh}")
    print(f"confidence_threshold:{confidence_threshold}")
    gt_faces = 0
    pred_faces = 0
    fp_info = []
    fn_info = []
    total_l1_loss = 0.0
    total_smooth_l1_loss = 0.0
    lmarks_cnt = 0

    tp = np.zeros(50*len(dataset))   ### todo: get total pred face cnt or set a very high number
    fp = np.zeros(50*len(dataset))

    if limit is None:
        num_imgs = len(dataset)
    else:
        num_imgs = limit
    for idx in tqdm(range(num_imgs)):     # for each img
        # for idx in tqdm(range(1)):
        target = dataset[idx][1]
        image_path = dataset_imgs[idx]
        # infer 1 image
        dets, img_raw = infer(net, image_path, img_dim, device, cfg, confidence_threshold, nms_threshold)

        predictions = np.array(sorted(dets, key=lambda x: x[4], reverse=True))
        gt = np.asarray(target, dtype=float)
        if closest_face_only:
            # gt = np.expand_dims(sorted(gt, key=lambda x: area(x), reverse=True)[0], axis=0)
            gt = np.expand_dims(sorted(gt, key=lambda x: get_centroid_distance(img_raw, x), reverse=False)[0], axis=0) # False: ascending
        gt_checked =  np.zeros(len(gt))  # gt flags for each face

        # calculate mAP
        if len(predictions) > 0:
            gt_t = torch.from_numpy(gt)
            pred_t = torch.from_numpy(predictions)
            if closest_face_only:
                # predictions = np.expand_dims(sorted(predictions, key=lambda x: area(x), reverse=True)[0], axis=0)
                predictions = np.expand_dims(sorted(predictions, key=lambda x: get_centroid_distance(img_raw, x), reverse=False)[0], axis=0) # False: ascending

            all_overlaps = jaccard(pred_t[:,:4], gt_t[:, :4]).cpu().numpy()
            for i in range(len(predictions)):   # for each detected face
                overlaps = all_overlaps[i]
                # print(overlaps)
                max_overlap = np.max(overlaps)
                jmax = np.argmax(overlaps)

                if max_overlap >= iou_thresh:
                    if gt_checked[jmax] == 0:
                        tp[pred_faces+i] = 1.0
                        gt_checked[jmax] = 1

                        if gt_t[jmax][-1].item() == 1.0:  # if this bbx used for lmarks eval
                            with torch.no_grad():
                                lmarks_cnt += 10  # 10 landmarks data point
                                total_l1_loss += torch.nn.L1Loss()( pred_t[i][5:15], gt_t[jmax][4:14]).item()
                                total_smooth_l1_loss += F.smooth_l1_loss(pred_t[i][5:15], gt_t[jmax][4:14]).item()
                    else:
                        fp[pred_faces+i] = 1.0
                        # Save FP information
                        fp_info.append({'image_idx': idx, 'image_path':dataset_imgs[idx] , 'fp_idx': i, 'fp_box': predictions[i, :4]})
                else:
                    fp[pred_faces+i] = 1.0
                    # Save FP information
                    fp_info.append({'image_idx': idx, 'image_path':dataset_imgs[idx], 'fp_idx': i, 'fp_box': predictions[i, :4]})

        # Find FNs
        for j in range(len(gt)):
            if gt_checked[j] == 0:
                # Save FN information
                fn_info.append({'image_idx': idx, 'image_path':dataset_imgs[idx] , 'fn_idx': j, 'fn_box': gt[j, :]})

        # update face counts
        gt_faces += len(gt)
        pred_faces += len(predictions)

    fp = fp[0:pred_faces]
    tp = tp[0:pred_faces]
    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)

    recalls = tp / float(gt_faces)

    # avoid divide by zero in case the first detection matches a difficult ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    ap = get_ap(recalls, precisions)

    print("Total GT Face:",gt_faces)
    print("Total detected Face:",pred_faces)
    print("Total landmarks evaluated:", lmarks_cnt)
    print("==================== Results ====================")
    if pred_faces > 0:
        print("precision:",precisions[-1])
        print("recall:", recalls[-1])
        print("lmark l1 loss:", total_l1_loss/lmarks_cnt)
        print("lmark smoothed l1 loss:", total_smooth_l1_loss/lmarks_cnt)
    print("Val AP: {}".format(ap))
    print("fps:", len(fp_info))
    print("fns", len(fn_info))
    print("=================================================")

    if log_file_path is not None:
        if epoch is None:
            epoch = 0
        if pred_faces > 0:
            log_entry = f"[{mode}] epoch {epoch}, mAP: {ap}, precision: {precisions[-1]}, recall: {recalls[-1]}, fps:{len(fp_info)}, fns:{len(fn_info)}, lmark_l1_loss:{total_l1_loss/lmarks_cnt}, lmark_smoothed_l1_loss:{total_smooth_l1_loss/lmarks_cnt} ,infer_img_dim: {img_dim}, iou_thresh: {iou_thresh} , confidence_threshold:{confidence_threshold}"
        else:
            log_entry = f"[{mode}] epoch {epoch}, mAP: {ap}, infer_img_dim: {img_dim}, iou_thresh: {iou_thresh}, confidence_threshold:{confidence_threshold}"
        write_to_log(log_file_path, log_entry)
    if save_fp_fn:
        write_fpfn_to_csv(fp_info, fn_info, "./results/", fp_output_file , fn_output_file)
    return  ap

def write_to_log(filepath:str, entry:str):
    log_file = open(filepath, 'a' if os.path.isfile(filepath) else 'w')
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry  = timestamp + "|" +entry + "\n"
    log_file.write(log_entry)
    log_file.close()

def write_fpfn_to_csv(fp_info, fn_info, result_folder, fp_output_file , fn_output_file):
    with open(result_folder+fp_output_file, 'w', newline='') as csvfile:
        fieldnames = ['image_idx', 'image_path', 'fp_idx', 'fp_box']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fp_info)

    with open(result_folder+fn_output_file, 'w', newline='') as csvfile:
        fieldnames = ['image_idx','image_path' , 'fn_idx', 'fn_box']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fn_info)

    print("FP/FN information saved to", result_folder+fp_output_file)
    print("=================================================")

def detect(mode, data_folder_path, label_files, cfg, net=None, net_ckpt_path=None , log_file_path=None, limit=None, confidence_threshold=None, nms_threshold=None):

    write_to_log(log_file_path, f"[{mode}], dataset: {label_files}")

    # load trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if net is None:
        print("No model being passed, reading from checkpoints:", net_ckpt_path)
        net = load_model(cfg, mode, device, model_weight_path=net_ckpt_path)
    else:
        print("Evaluating passed model for " + mode)
        net = net.eval()

    # load data
    dataset = FaceDetection_vol1(label_folder_path=data_folder_path, label_files=label_files, preproc=None)
    dataset_imgs = dataset.imgs_path

    # constants
    if confidence_threshold is None:
        confidence_threshold = cfg['infer_confidence_threshold']
    if nms_threshold is None:
        nms_threshold = cfg['infer_nms_threshold']  # 0.4
    iou_thresh = cfg['infer_iou_thresh']  # 0.5
    img_dim = cfg['infer_image_size']
    print(f"img_dim:{str(img_dim)}")
    print(f"iou_thresh:{iou_thresh}")
    print(f"confidence_threshold:{confidence_threshold}")

    if limit is None:
        num_imgs = len(dataset)
    else:
        num_imgs = limit

    for idx in tqdm(range(num_imgs)):     # for each img
        # target = dataset[idx][1]
        image_path = dataset_imgs[idx]

        # infer 1 image
        dets, img_raw = infer(net, image_path, img_dim, device, cfg, confidence_threshold, nms_threshold)
        predictions = np.array(sorted(dets, key=lambda x: x[4], reverse=True))
        
        bbox = []

        res_file_path = cfg['label_output_file']

        output_label(predictions, image_path, res_file_path)

        if cfg['save_image'] and idx < 100:
            save_image(dets, cfg, img_raw, idx)

    print('finish writing labels to output file!')

if __name__ == '__main__':
    cfg = cfg_mnet

    # dataset mode
    mode = 'test'
    # print("mode:", mode)

    # data_folder_path = f"/home/jovyan/data/vol_3/face_detection_files/face_detection_v1_1/data/{mode}/"
    # data_folder_path = f"{cfg['data_folder']}/{mode}/"

    # data_folder_path = f"/home/jovyan/data/vol_1/liveness_files/selfie/"
    data_folder_path = f"{cfg['data_folder']}"

    # label_files = ['label_aurora.txt', 'label_driver.txt', 'label_consumer.txt', 'label_retina_fail.txt']
    label_files = ['eye_label_new.csv']

    print("label_files", label_files)

    # log for detect
    logfile_detect = "model_best_detect.txt"

    # finetune model checkpoint
    net_ckpt_path = './weights/mobilenet0.25_best.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = load_model(cfg, mode, device, model_weight_path=net_ckpt_path)
    print("Using model: ", net_ckpt_path)

    detect(mode, data_folder_path, label_files, cfg, net=net, log_file_path=logfile_detect, confidence_threshold=0.1)
