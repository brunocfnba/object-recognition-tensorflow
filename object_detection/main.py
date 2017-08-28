import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image

sys.path.append("..")
from object_detection.utils import label_map_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

# This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

detection_graph = tf.Graph()


def load_to_memory():
    global loaded
    global detection_graph

    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    loaded = True

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


IMAGE_SIZE = (20, 16)

loaded = False


def detect_objects(image, threshold):
    global detection_graph
    global loaded
    if not loaded:
        print("here")
        load_to_memory()
        print(loaded)

    image_directory = 'test_images'
    image_path = os.path.join(image_directory, '{}'.format(image))

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_path = image_path
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)

            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            obj_above_thres = 0
            objs_found = []
            for i in scores[0]:
                if i >= threshold:
                    print(category_index[classes[0][obj_above_thres]])
                    objs_found.append({'name': category_index[classes[0][obj_above_thres]]['name'],'confidence': str(i) })
                    obj_above_thres = obj_above_thres + 1

            return objs_found

# print(detect_objects('image1.jpg', 0.50))
