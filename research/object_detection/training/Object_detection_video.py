######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a video.
# It draws boxes, scores, and labels around the objects of interest in each
# frame of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import argparse
from glob import glob
import skimage.io as skio

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


def directory_iterator(input_dir):
    files = glob(os.path.join(input_dir,'*.jpg'))
    for file in files:
        yield file, skio.imread(file)



def process_video(model_path, video_path, labelmap_path, threshold, out_video_path, subsample, offset):

    label_map = label_map_util.load_labelmap(labelmap_path)
    num_classes = 1
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    # Tensor outputs
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Open video files
    video = cv2.VideoCapture(video_path)
    imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if out_video_path is not None:
        out_video = cv2.VideoWriter(out_video_path, -1, 20.0, (int(imW // subsample), int(imH // subsample)))

    idx = 0
    while video.isOpened():
        ret, frame = video.read()
        idx += 1
        print("\r{}".format(idx), end='')
        if idx < offset:
            continue
        if frame is None:
            print("None frame!?")
            break
        if subsample > 1:
            imH, imW, _ = frame.shape
            frame = cv2.resize(frame, (imW // subsample, imH // subsample))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visualise the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=threshold)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector (press q to quit)', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        if out_video_path is not None:
            out_video.write(frame)

    # Clean up
    video.release()
    if out_video_path is not None:
        out_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply frozen inference graph to a video')
    parser.add_argument('-m', '--model', type=str, help="Path to frozen inference graph")
    parser.add_argument('-v', '--video', type=str, help="Path to video to process")
    parser.add_argument('-l', '--labelmap', type=str, help="Path to label map")
    parser.add_argument('-s', '--subsample', type=int, default=1, help="Subsample the image (make smalled by this factor)")
    parser.add_argument('-o', '--output', type=str, default=None, help="Save results to this video")
    parser.add_argument('--offset', type=int, default=0, help='Offset into file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold')

    args = parser.parse_args()
    print(args)

    process_video(args.model, args.video, args.labelmap, args.threshold, args.output, args.subsample, args.offset)
