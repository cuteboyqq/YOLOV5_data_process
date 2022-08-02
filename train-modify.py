from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
#from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all
import tqdm
from tensorflow.python.saved_model import tag_constants
import cv2
from core.yolov4 import filter_boxes
import glob
import json
flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_string('weights', './checkpoints_yolov4_20220729_ciou_tf25_mosaic_aug/yolov4', 'pretrained weights')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def main(_argv):
    
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #if len(physical_devices) > 0:
        #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    from core.yolov4 import YOLO, decode, compute_loss, decode_train
    trainset = Dataset(FLAGS, is_training=True)
    testset = Dataset(FLAGS, is_training=False)
    logdir = "./data/log"
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    # train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)

    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    if FLAGS.tiny:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()

    if FLAGS.weights == None:
        print("Training from scratch")
    else:
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
        else:
            model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)


    optimizer = tf.keras.optimizers.Adam()
    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

    # define training step function
    # @tf.function
    def train_step(image_data, target,epoch):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            #tf.print("=>EPOCH: %4d STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
            #         "prob_loss: %4.2f   total_loss: %4.2f" % (epoch,global_steps, total_steps, optimizer.lr.numpy(),
            #                                                   giou_loss, conf_loss,
            #                                                   prob_loss, total_loss))
            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
            else:
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
            
            return total_loss,giou_loss,conf_loss,prob_loss
    

    def test_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            #tf.print("=>TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
            #         "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
            #                                                   prob_loss, total_loss))
            return total_loss,giou_loss,conf_loss,prob_loss
    
    
        
    def Validation(weights,INPUT_SIZE,framework,annotation_path,model,tiny,IOU,SCORE):
        #INPUT_SIZE = FLAGS.size
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)

        predicted_dir_path = '.\mAP\predicted'
        ground_truth_dir_path = '.\mAP\ground-truth'
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        #if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)
        #os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

        # Build Model
        if framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        else:
            saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']

        num_lines = sum(1 for line in open(annotation_path))
        with open(annotation_path, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

                if len(bbox_data_gt) == 0:
                    bboxes_gt = []
                    classes_gt = []
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

                print('=> ground truth of %s:' % image_name)
                num_bbox_gt = len(bboxes_gt)
                with open(ground_truth_path, 'w') as f:
                    for i in range(num_bbox_gt):
                        class_name = CLASSES[classes_gt[i]]
                        xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                        bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())
                print('=> predict result of %s:' % image_name)
                predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
                # Predict Process
                image_size = image.shape[:2]
                # image_data = utils.image_preprocess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
                image_data = cv2.resize(np.copy(image), (INPUT_SIZE, INPUT_SIZE))
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)

                if framework == 'tflite':
                    interpreter.set_tensor(input_details[0]['index'], image_data)
                    interpreter.invoke()
                    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                    if model == 'yolov4' and tiny == True:
                        boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25)
                    else:
                        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25)
                else:
                    batch_data = tf.constant(image_data)
                    pred_bbox = infer(batch_data)
                    for key, value in pred_bbox.items():
                        boxes = value[:, :, 0:4]
                        pred_conf = value[:, :, 4:]

                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=IOU,
                    score_threshold=SCORE
                )
                boxes, scores, classes, valid_detections = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

                # if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
                #     image_result = utils.draw_bbox(np.copy(image), [boxes, scores, classes, valid_detections])
                #     cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH + image_name, image_result)

                with open(predict_result_path, 'w') as f:
                    image_h, image_w, _ = image.shape
                    for i in range(valid_detections[0]):
                        if int(classes[0][i]) < 0 or int(classes[0][i]) > NUM_CLASS: continue
                        coor = boxes[0][i]
                        coor[0] = int(coor[0] * image_h)
                        coor[2] = int(coor[2] * image_h)
                        coor[1] = int(coor[1] * image_w)
                        coor[3] = int(coor[3] * image_w)

                        score = scores[0][i]
                        class_ind = int(classes[0][i])
                        class_name = CLASSES[class_ind]
                        score = '%.4f' % score
                        ymin, xmin, ymax, xmax = list(map(str, coor))
                        bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())
                print(num, num_lines)
                
    
    def precision_recall_mAP(output,draw_plot=True,show_animation=True,ignore=[],set_class_iou=[],MINOVERLAP=0.5,quiet=True,no_plot=False):
        
        img_path = 'images'
        
        specific_iou_flagged = False
        if set_class_iou is not None:
            specific_iou_flagged = True
        
        # try to import Matplotlib if the user didn't choose the option --no-plot
        draw_plot = False
        if not no_plot:
          try:
            import matplotlib.pyplot as plt
            draw_plot = True
          except ImportError:
            print("\"matplotlib\" not found, please install it to get the resulting plots.")
            no_plot = True
            
            
      
        """
         Convert the lines of a file to a list
        """
        def file_lines_to_list(path):
          # open txt file lines to a list
          with open(path) as f:
            content = f.readlines()
          # remove whitespace characters like `\n` at the end of each line
          content = [x.strip() for x in content]
          return content
        """
         check if the number is a float between 0.0 and 1.0
        """
        def is_float_between_0_and_1(value):
          try:
            val = float(value)
            if val > 0.0 and val < 1.0:
              return True
            else:
              return False
          except ValueError:
            return False
        """
         Draws text in image
        """
        def draw_text_in_image(img, text, pos, color, line_width):
          font = cv2.FONT_HERSHEY_PLAIN
          fontScale = 1
          lineType = 1
          bottomLeftCornerOfText = pos
          cv2.putText(img, text,
              bottomLeftCornerOfText,
              font,
              fontScale,
              color,
              lineType)
          text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
          return img, (line_width + text_width)
        """
         Calculate the AP given the recall and precision array
          1st) We compute a version of the measured precision/recall curve with
               precision monotonically decreasing
          2nd) We compute the AP as the area under this curve by numerical integration.
        """
        def voc_ap(rec, prec):
          """
          --- Official matlab code VOC2012---
          mrec=[0 ; rec ; 1];
          mpre=[0 ; prec ; 0];
          for i=numel(mpre)-1:-1:1
              mpre(i)=max(mpre(i),mpre(i+1));
          end
          i=find(mrec(2:end)~=mrec(1:end-1))+1;
          ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
          """
          rec.insert(0, 0.0) # insert 0.0 at begining of list
          rec.append(1.0) # insert 1.0 at end of list
          mrec = rec[:]
          prec.insert(0, 0.0) # insert 0.0 at begining of list
          prec.append(0.0) # insert 0.0 at end of list
          mpre = prec[:]
          """
           This part makes the precision monotonically decreasing
            (goes from the end to the beginning)
            matlab:  for i=numel(mpre)-1:-1:1
                        mpre(i)=max(mpre(i),mpre(i+1));
          """
          # matlab indexes start in 1 but python in 0, so I have to do:
          #   range(start=(len(mpre) - 2), end=0, step=-1)
          # also the python function range excludes the end, resulting in:
          #   range(start=(len(mpre) - 2), end=-1, step=-1)
          for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])
          """
           This part creates a list of indexes where the recall changes
            matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
          """
          i_list = []
          for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
              i_list.append(i) # if it was matlab would be i + 1
          """
           The Average Precision (AP) is the area under the curve
            (numerical integration)
            matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
          """
          ap = 0.0
          for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
          return ap, mrec, mpre
        """
         Create a "tmp_files/" and "results/" directory
        """
        tmp_files_path = "tmp_files"
        if not os.path.exists(tmp_files_path): # if it doesn't exist already
          os.makedirs(tmp_files_path)
        results_files_path = output
        if os.path.exists(results_files_path): # if it exist already
          # reset the results directory
          shutil.rmtree(results_files_path)
        
        os.makedirs(results_files_path)
        if draw_plot:
          os.makedirs(results_files_path + "/classes")
        if show_animation:
          os.makedirs(results_files_path + "/images")
          os.makedirs(results_files_path + "/images/single_predictions")
        
        """
         Ground-Truth
           Load each of the ground-truth files into a temporary ".json" file.
           Create a list of all the class names present in the ground-truth (gt_classes).
        """
        # get a list with the ground-truth files
        ground_truth_files_list = glob.glob('./mAP/ground-truth/*.txt')
        if len(ground_truth_files_list) == 0:
          print("Error: No ground-truth files found!")
        ground_truth_files_list.sort()
        # dictionary with counter per class
        gt_counter_per_class = {}
        
        for txt_file in ground_truth_files_list:
          print(txt_file)
          file_id = txt_file.split(".txt",1)[0]
          file_id = os.path.basename(os.path.normpath(file_id))
          # check if there is a correspondent predicted objects file
          if not os.path.exists('./mAP/predicted/' + file_id + ".txt"):
            error_msg = "Error. File not found: predicted/" +  file_id + ".txt\n"
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
            print(error_msg)
          lines_list = file_lines_to_list(txt_file)
          # create ground-truth dictionary
          bounding_boxes = []
          is_difficult = False
          for line in lines_list:
            try:
              if "difficult" in line:
                  class_name, left, top, right, bottom, _difficult = line.split()
                  is_difficult = True
              else:
                  class_name, left, top, right, bottom = line.split()
            except ValueError:
              error_msg = "Error: File " + txt_file + " in the wrong format.\n"
              error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
              error_msg += " Received: " + line
              error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
              error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
              print(error_msg)
            # check if class is in the ignore list, if yes skip
            if class_name in ignore:
              continue
            bbox = left + " " + top + " " + right + " " +bottom
            if is_difficult:
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
                # count that object
                if class_name in gt_counter_per_class:
                  gt_counter_per_class[class_name] += 1
                else:
                  # if class didn't exist yet
                  gt_counter_per_class[class_name] = 1
          # dump bounding_boxes into a ".json" file
          with open(tmp_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)
        
        gt_classes = list(gt_counter_per_class.keys())
        # let's sort the classes alphabetically
        gt_classes = sorted(gt_classes)
        n_classes = len(gt_classes)
        #print(gt_classes)
        #print(gt_counter_per_class)
        
        """
         Check format of the flag --set-class-iou (if used)
          e.g. check if class exists
        """
        
        
        
        if specific_iou_flagged:
          n_args = len(set_class_iou)
          error_msg = \
            '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
          if n_args % 2 != 0:
            print('Error, missing arguments. Flag usage:' + error_msg)
          # [class_1] [IoU_1] [class_2] [IoU_2]
          # specific_iou_classes = ['class_1', 'class_2']
          specific_iou_classes = set_class_iou[::2] # even
          # iou_list = ['IoU_1', 'IoU_2']
          iou_list = set_class_iou[1::2] # odd
          if len(specific_iou_classes) != len(iou_list):
            print('Error, missing arguments. Flag usage:' + error_msg)
          for tmp_class in specific_iou_classes:
            if tmp_class not in gt_classes:
                  print('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
          for num in iou_list:
            if not is_float_between_0_and_1(num):
              print('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)
        
        """
         Predicted
           Load each of the predicted files into a temporary ".json" file.
        """
        # get a list with the predicted files
        predicted_files_list = glob.glob('./mAP/predicted/*.txt')
        predicted_files_list.sort()
        
        for class_index, class_name in enumerate(gt_classes):
          bounding_boxes = []
          for txt_file in predicted_files_list:
            print(txt_file)
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = txt_file.split(".txt",1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            if class_index == 0:
              if not os.path.exists('./mAP/ground-truth/' + file_id + ".txt"):
                error_msg = "Error. File not found: ground-truth/" +  file_id + ".txt\n"
                error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
                print(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
              try:
                tmp_class_name, confidence, left, top, right, bottom = line.split()
                #print(tmp_class_name, confidence, left, top, right, bottom)
              except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                error_msg += " Received: " + line
                print(error_msg)
              if tmp_class_name == class_name:
                #print("match")
                bbox = left + " " + top + " " + right + " " +bottom
                bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
                print(bounding_boxes)
          # sort predictions by decreasing confidence
          bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
          with open(tmp_files_path + "/" + class_name + "_predictions.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)
        
        """
         Calculate the AP for each class
        """
        sum_AP = 0.0
        ap_dictionary = {}
        # open file to store the results
        with open(results_files_path + "/results.txt", 'w') as results_file:
          results_file.write("# AP and precision/recall per class\n")
          count_true_positives = {}
          for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            """
             Load predictions of that class
            """
            predictions_file = tmp_files_path + "/" + class_name + "_predictions.json"
            predictions_data = json.load(open(predictions_file))
        
            """
             Assign predictions to ground truth objects
            """
            nd = len(predictions_data)
            tp = [0] * nd # creates an array of zeros of size nd
            fp = [0] * nd
            for idx, prediction in enumerate(predictions_data):
              file_id = prediction["file_id"]
              if show_animation:
                # find ground truth image
                ground_truth_img = glob.glob1(img_path, file_id + ".*")
                #tifCounter = len(glob.glob1(myPath,"*.tif"))
                if len(ground_truth_img) == 0:
                  print("Error. Image not found with id: " + file_id)
                elif len(ground_truth_img) > 1:
                  print("Error. Multiple image with id: " + file_id)
                else: # found image
                  #print(img_path + "/" + ground_truth_img[0])
                  # Load image
                  img = cv2.imread(img_path + "/" + ground_truth_img[0])
                  # load image with draws of multiple detections
                  img_cumulative_path = results_files_path + "/images/" + ground_truth_img[0]
                  if os.path.isfile(img_cumulative_path):
                    img_cumulative = cv2.imread(img_cumulative_path)
                  else:
                    img_cumulative = img.copy()
                  # Add bottom border to image
                  bottom_border = 60
                  BLACK = [0, 0, 0]
                  img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
              # assign prediction to ground truth object if any
              #   open ground-truth with that file_id
              gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
              ground_truth_data = json.load(open(gt_file))
              ovmax = -1
              gt_match = -1
              # load prediction bounding-box
              bb = [ float(x) for x in prediction["bbox"].split() ]
              for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                  bbgt = [ float(x) for x in obj["bbox"].split() ]
                  bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                  iw = bi[2] - bi[0] + 1
                  ih = bi[3] - bi[1] + 1
                  if iw > 0 and ih > 0:
                    # compute overlap (IoU) = area of intersection / area of union
                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                            + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                    ov = iw * ih / ua
                    if ov > ovmax:
                      ovmax = ov
                      gt_match = obj
        
              # assign prediction as true positive/don't care/false positive
              if show_animation:
                status = "NO MATCH FOUND!" # status is only used in the animation
              # set minimum overlap
              min_overlap = MINOVERLAP
              if specific_iou_flagged:
                if class_name in specific_iou_classes:
                  index = specific_iou_classes.index(class_name)
                  min_overlap = float(iou_list[index])
              if ovmax >= min_overlap:
                if "difficult" not in gt_match:
                    if not bool(gt_match["used"]):
                      # true positive
                      tp[idx] = 1
                      gt_match["used"] = True
                      count_true_positives[class_name] += 1
                      # update the ".json" file
                      with open(gt_file, 'w') as f:
                          f.write(json.dumps(ground_truth_data))
                      if show_animation:
                        status = "MATCH!"
                    else:
                      # false positive (multiple detection)
                      fp[idx] = 1
                      if show_animation:
                        status = "REPEATED MATCH!"
              else:
                # false positive
                fp[idx] = 1
                if ovmax > 0:
                  status = "INSUFFICIENT OVERLAP"
        
              """
               Draw image to show animation
              """
              if show_animation:
                height, widht = img.shape[:2]
                # colors (OpenCV works with BGR)
                white = (255,255,255)
                light_blue = (255,200,100)
                green = (0,255,0)
                light_red = (30,30,255)
                # 1st line
                margin = 10
                v_pos = int(height - margin - (bottom_border / 2))
                text = "Image: " + ground_truth_img[0] + " "
                img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
                if ovmax != -1:
                  color = light_red
                  if status == "INSUFFICIENT OVERLAP":
                    text = "IoU: {0:.2f}% ".format(ovmax*100) + "< {0:.2f}% ".format(min_overlap*100)
                  else:
                    text = "IoU: {0:.2f}% ".format(ovmax*100) + ">= {0:.2f}% ".format(min_overlap*100)
                    color = green
                  img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                # 2nd line
                v_pos += int(bottom_border / 2)
                rank_pos = str(idx+1) # rank position (idx starts at 0)
                text = "Prediction #rank: " + rank_pos + " confidence: {0:.2f}% ".format(float(prediction["confidence"])*100)
                img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                color = light_red
                if status == "MATCH!":
                  color = green
                text = "Result: " + status + " "
                img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
        
                font = cv2.FONT_HERSHEY_SIMPLEX
                if ovmax > 0: # if there is intersections between the bounding-boxes
                  bbgt = [ int(x) for x in gt_match["bbox"].split() ]
                  cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
                  cv2.rectangle(img_cumulative,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
                  cv2.putText(img_cumulative, class_name, (bbgt[0],bbgt[1] - 5), font, 0.6, light_blue, 1, cv2.LINE_AA)
                bb = [int(i) for i in bb]
                cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                cv2.rectangle(img_cumulative,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                cv2.putText(img_cumulative, class_name, (bb[0],bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
                # show image
                cv2.imshow("Animation", img)
                cv2.waitKey(20) # show for 20 ms
                # save image to results
                output_img_path = results_files_path + "/images/single_predictions/" + class_name + "_prediction" + str(idx) + ".jpg"
                cv2.imwrite(output_img_path, img)
                # save the image with all the objects drawn to it
                cv2.imwrite(img_cumulative_path, img_cumulative)
        
            #print(tp)
            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
              fp[idx] += cumsum
              cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
              tp[idx] += cumsum
              cumsum += val
            #print(tp)
            rec = tp[:]
            for idx, val in enumerate(tp):
              rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            #print(rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
              prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            #print(prec)
        
            ap, mrec, mprec = voc_ap(rec, prec)
            print('ap, mrec, mprec = {}, {}, {}'.format(ap, mrec, mprec))
           
            sum_AP += ap
            text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP  " #class_name + " AP = {0:.2f}%".format(ap*100)
            """
             Write to results.txt
            """
            rounded_prec = [ '%.2f' % elem for elem in prec ]
            rounded_rec = [ '%.2f' % elem for elem in rec ]
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")
            if not quiet:
              print(text)
            ap_dictionary[class_name] = ap
        
            """
             Draw plot
            """
            if draw_plot:
              plt.plot(rec, prec, '-o')
              # add a new penultimate point to the list (mrec[-2], 0.0)
              # since the last line segment (and respective area) do not affect the AP value
              area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
              area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
              plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
              # set window title
              fig = plt.gcf() # gcf - get current figure
              fig.canvas.set_window_title('AP ' + class_name)
              # set plot title
              plt.title('class: ' + text)
              #plt.suptitle('This is a somewhat long figure title', fontsize=16)
              # set axis titles
              plt.xlabel('Recall')
              plt.ylabel('Precision')
              # optional - set axes
              axes = plt.gca() # gca - get current axes
              axes.set_xlim([0.0,1.0])
              axes.set_ylim([0.0,1.05]) # .05 to give some extra space
              # Alternative option -> wait for button to be pressed
              #while not plt.waitforbuttonpress(): pass # wait for key display
              # Alternative option -> normal display
              #plt.show()
              # save the plot
              fig.savefig(results_files_path + "/classes/" + class_name + ".png")
              plt.cla() # clear axes for next plot
        
          if show_animation:
            cv2.destroyAllWindows()
        
          results_file.write("\n# mAP of all classes\n")
          mAP = sum_AP / n_classes
          text = "mAP = {0:.2f}%".format(mAP*100)
          results_file.write(text + "\n")
          print(text)
        
        # remove the tmp_files directory
        shutil.rmtree(tmp_files_path)
        
    records = []
    VAL_LOSS = 100000
    for epoch in range(first_stage_epochs + second_stage_epochs):
        if epoch < first_stage_epochs:
            if not isfreeze:
                isfreeze = True
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    freeze_all(freeze)
        elif epoch >= first_stage_epochs:
            if isfreeze:
                isfreeze = False
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    unfreeze_all(freeze)
        
        pbar_train = tqdm.tqdm(trainset)
        pbar_test = tqdm.tqdm(testset)
        print(' Train Epoch  Total_loss  giou_loss  conf_loss  prob_loss')
        print('==========================================================')
        Total_Train_Loss, Total_giou_loss, Total_conf_loss, Total_prob_loss = 0,0,0,0
        for image_data, target in pbar_train:
            total_loss_train,giou_loss,conf_loss,prob_loss = train_step(image_data, target, epoch)
            
            total_loss_train =  float(int(total_loss_train*100)/100.0)
            giou_loss =  float(int(giou_loss.numpy()*100)/100.0)
            conf_loss =  float(int(conf_loss.numpy()*100)/100.0)
            prob_loss =  float(int(prob_loss.numpy()*100)/100.0)
            
            
            
            
            bar_str =   '     '+str(epoch+1) + '         '+ str(total_loss_train)\
                      + '     ' + str(giou_loss)\
                      + '     ' + str(conf_loss)\
                      + '     ' + str(prob_loss)
            PREFIX = colorstr(bar_str)
            pbar_train.desc = f'{PREFIX}'
            
            Total_Train_Loss+=total_loss_train
            Total_giou_loss+=giou_loss
            Total_conf_loss+=conf_loss
            Total_prob_loss+=prob_loss
            
        records.append(['Train', epoch+1, Total_Train_Loss, Total_giou_loss, Total_conf_loss, Total_prob_loss])
        DO_VAL = True
        save_valloss_min_model = True
        Total_Val_Loss, Total_val_giou_loss, Total_val_conf_loss, Total_val_prob_loss = 0,0,0,0
        if DO_VAL:
            print('      Val Epoch  Total_loss  giou_loss  conf_loss  prob_loss')
            print('     --------------------------------------------------------')
            save_valloss_min_model = False
            for image_data, target in pbar_test:
                total_loss_val,giou_loss,conf_loss,prob_loss = test_step(image_data, target)
                
                total_loss_val =  float(int(total_loss_val*100)/100.0)
                giou_loss =  float(int(giou_loss.numpy()*100)/100.0)
                conf_loss =  float(int(conf_loss.numpy()*100)/100.0)
                prob_loss =  float(int(prob_loss.numpy()*100)/100.0)
                
                bar_str =   '          '+str(epoch+1) + '         '+ str(total_loss_val)\
                          + '     ' + str(giou_loss)\
                          + '     ' + str(conf_loss)\
                          + '     ' + str(prob_loss)
                PREFIX = colorstr(bar_str)
                pbar_test.desc = f'{PREFIX}'
                
                Total_Val_Loss+=total_loss_train
                Total_val_giou_loss+=giou_loss
                Total_val_conf_loss+=conf_loss
                Total_val_prob_loss+=prob_loss
                
            
            if Total_Val_Loss < VAL_LOSS:
                VAL_LOSS = Total_Val_Loss
                save_valloss_min_model = True
            records.append(['Val  ', epoch+1, Total_Val_Loss, Total_val_giou_loss, Total_val_conf_loss, Total_val_prob_loss])
        if save_valloss_min_model:
            print('Val loss: {}, start to save model'.format(VAL_LOSS))
            #tf.saved_model.save(model, './model')
            model.save_weights("./checkpoints_yolov4_20220729_ciou_tf25_mosaic_aug/yolov4")
            #model.save('./model_20220731')
    import csv
    result_path = './train/checkpoints_yolov4_20220729_ciou_tf25_mosaic_aug.csv'
    fields = ['data', 'Epoch', 'Total_loss', 'giou_loss', 'conf_loss', 'prob_loss']
    with open(result_path, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(records)
        #annotation_path=r'C:\YOLOV4-TF\datasets\factory_data_val_small.txt'
        #Validation('./checkpoints_20220728/yolov4-416',INPUT_SIZE=416,framework='tf',annotation_path=annotation_path,model='yolov4',tiny=False,IOU=0.45,SCORE=0.5)
        #output = 'mAP/results'
        #precision_recall_mAP(output,draw_plot=True,show_animation=False,ignore=[],set_class_iou=[],MINOVERLAP=0.5,quiet=False,no_plot=False)
        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass