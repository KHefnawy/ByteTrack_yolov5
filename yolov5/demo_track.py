from loguru import logger
import numpy as np

import cv2

import torch


from utils.visualize import plot_tracking
from tracker.byte_tracker import BYTETracker
from tracking_utils.timer import Timer
from utils.boxes import postprocess
import argparse
import os
import time
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.augmentations import preproc
from utils.general import non_max_suppression


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
#device = "cpu"


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./datasets/mot/train/MOT17-02-FRCNN/img1", help="path to images or video"
        # "--path", default="./videos/16h-17h.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="test",
        help="save name for results txt/video",
    )

    parser.add_argument("-c", "--ckpt", default="weights/crowdhuman_yolov5m.pt", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--num_classes", type=int, default=1, help="number of classes")
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=(1280,1280), type=tuple, help="test image size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=5, help='filter out tiny boxes')
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=5,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    return parser

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        # exp,
        num_classes, conf_thresh, nms_thresh, test_size,device,
        trt_file=None,
        
        fp16=False
    ):
        self.model = model
        self.num_classes = num_classes # 1
        self.confthre = conf_thresh
        self.nmsthre = nms_thresh
        self.test_size = test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            ##change here 
            if device=="gpu":
              x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            else:
              x = torch.ones(1, 3, test_size[0], test_size[1])
            self.model(x)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
           
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

      
        #img = letterbox(img, self.test_size)[0]
        img = preproc(img, self.test_size,mean=None, std=None)[0]
        #img = letterbox(img)[0]
        #print("img_shape after letterbox =",img2.shape)
       # print("img_shape after letterbox =",img.shape)

        # Convert
        #img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
       # img = np.ascontiguousarray(img)
        ###change here to use cpu instead of gpud
        if self.device=="gpu":
          img = torch.from_numpy(img).cuda()
        else:
          img = torch.from_numpy(img)
        img = img.half() if self.fp16 else img.float()  # uint8 to fp16/32
        #img /= 255  # 0 - 255 to 0.0 - 1.0
        # print(img)
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        with torch.no_grad():
            timer.tic()
            # print('image shape:', img.size())
            #print("infernec ",img)
            outputs = self.model(img, False, False)
            #print("outputs",outputs[0])
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
           
            timer.toc()
           
        return outputs, img_info

def save_outputs(outputs, folder, save_name):
    sn = save_name.split('/')[-1].replace('.jpg', '.txt')
    # if not os.path.exists('yolov5_outputs'):
    #     os.mkdir('yolov5_outputs')
    
    sn = os.path.join('runs', folder, sn)
    print("sn ",sn)
    # if not os.path.exists(os.path.join('yolov5_outputs', folder)):
    #     os.mkdir(os.path.join('yolov5_outputs', folder))
    # open or creat new file if dont exist
    with open(sn, 'w') as f:
        if outputs[0] is not None:
                for i in range(len(outputs[0])):
                    op = outputs[0][i].tolist()
                    for j in op:
                        f.write(str(j) + ' ')
                    f.write('\n')
            


def image_demo(predictor, vis_folder, path, current_time, save_result, save_name, test_size):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort(key=lambda x: int(x.split('/')[-1][:-4]))
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    for image_name in files:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        print("imagename :",image_name)    
        outputs, img_info = predictor.inference(image_name, timer)
        save_outputs(outputs, save_name, image_name)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], test_size)
            # print('height:', img_info['height'], 'width:', img_info['width'])
            # print('test size:', exp.test_size)
            print('online_targets:', len(online_targets))
            # print(online_targets)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            timer.toc()
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids,online_ids ,frame_id=frame_id + 1,
                                          fps=1. / timer.average_time)
        else:
            timer.toc()
            online_im = img_info['raw_img']

        #result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        save_result=True
        if save_result:
            save_folder = os.path.join(
                vis_folder, save_name
            )
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            print("Save tracked image to {}".format(save_file_name))
            cv2.imwrite(save_file_name, online_im)
        ch = cv2.waitKey(0)
        frame_id += 1
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    result_filename = os.path.join(vis_folder, os.path.basename(save_name + '.txt'))
    print("Save results to {}".format(result_filename))
    write_results(result_filename, results)


def imageflow_demo(predictor, vis_folder, current_time, args):
    import cv2
    print(args.path)
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps= ",fps)
    save_name = args.save_name
    save_folder = os.path.join(
        vis_folder, save_name
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, save_name + ".mp4")
        print("video save path is ",save_path)
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        
        
        #print("hereeeeeeeeeeeeeee",ret_val)
        
        if ret_val:
            ## CHANGE HERE !!!!!!!!  change frame size 
            #frame = cv2.resize(frame, (608, 1088))
           # print(frame)
            #print("inferneceeeeee")
            #frame = cv2.resize(frame, (1280, 720))
            outputs, img_info = predictor.inference(frame, timer)
            #print("outputs",outputs)
            #print("img_info",img_info)

            #print("outputs",outputs)

            # for i, det in enumerate(outputs):
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], args.tsize)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
                timer.toc()
                online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids,online_ids,frame_id=frame_id + 1,
                                          fps=1. / timer.average_time)
            else:
                timer.toc()
                online_im = img_info['raw_img']
            save_result=True
            #if args.save_result:
            if save_result:
                vid_writer.write(online_im)
                #print("hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
                #cv2.imshow("online_im", online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1


def main(args):
    # if not args.experiment_name:
    #     args.experiment_name = exp.exp_name

    file_name = os.path.join('runs', '')
    os.makedirs(file_name, exist_ok=True)
    vis_folder = os.path.join(file_name, "track")
    os.makedirs(vis_folder, exist_ok=True)
        

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    conf_thresh = args.conf
    nms_thresh = args.nms
    num_classes = args.num_classes

    ckpt_file = args.ckpt

    if args.device=="gpu":

       model = DetectMultiBackend(ckpt_file, device='cuda')
    else:
        model = DetectMultiBackend(ckpt_file, device='cpu')
    if args.fp16:
        model.model.half()
    model.eval()

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False 
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
    #print("siiiize",args.tsize)
    predictor = Predictor(model, num_classes, conf_thresh, nms_thresh, args.tsize,  args.device ,trt_file, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result, args.save_name, args.tsize)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    assert args.demo in ["image", "video", "webcam"], "demo type not supported, only support [image, video, webcam]"
    #assert args.tsize in [(800, 1440), (608, 1088)], "tsize not supported, only (800, 1440) and (608, 1088)"
    main(args)