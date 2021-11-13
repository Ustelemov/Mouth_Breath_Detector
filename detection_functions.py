import tensorflow as tf
import cv2
import numpy as np
import dlib

def shape_to_np(shape, dtype="int"):
  # initialize the list of (x, y)-coordinates
  coords = np.zeros((shape.num_parts, 2), dtype=dtype)

  # loop over all facial landmarks and convert them
  # to a 2-tuple of (x, y)-coordinates
  for i in range(0, shape.num_parts):
    coords[i] = (shape.part(i).x, shape.part(i).y)

  # return the list of (x, y)-coordinates
  return coords


class SSDFaceDetector():

    def __init__(self,model_path, input_width, input_height,det_threshold=0.3):

        self.det_threshold = det_threshold
        self.input_width = input_width
        self.input_height = input_height
        self.detection_graph = tf.Graph()


        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.compat.v1.Session(graph=self.detection_graph)

    
    def detect_face(self, image_before):

        h, w,c = image_before.shape

        image = image_before.copy()

        image_res = cv2.resize(image, (self.input_width, self.input_height))
        image_np = cv2.cvtColor(image_res, cv2.COLOR_BGR2RGB)

        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')

        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)

        filtered_score_index = np.argwhere(
            scores >= self.det_threshold).flatten()

        selected_boxes = boxes[filtered_score_index]

        faces = np.array([[
            int(x1 * w),
            int(y1 * h),
            int(x2 * w),
            int(y2 * h),
        ] for y1, x1, y2, x2 in selected_boxes])

        return faces

class DlibFaceAligner:
    def __init__(self,landmarks_path,model_path,SSD_input_w,SSD_input_h):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = dlib.shape_predictor(landmarks_path)

        self.detector = SSDFaceDetector(model_path,input_width = SSD_input_w, input_height = SSD_input_h)

    def get_lips_points(self,cv2_frame):
    
        coords = self.detector.detect_face(cv2_frame)

        if len(coords) == 0:
          return (False,0,0)
        else:
          x1, y1, x2, y2 = [int(c) for c in coords[0]]
          rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)

          # dlib works with gray, so as image we're waiting for bgr opencv captured image and convert that to
          gray = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2GRAY)


          # convert the landmark (x, y)-coordinates to a NumPy array
          shape = self.predictor(gray, rect)
          shape = shape_to_np(shape)

          bottom_lip_highest_y = min(y for x,y in shape[65:67])
          top_lip_lowest_y = max(y for x,y in shape[61:63])

        #   bottom_lip_highest_y = shape[66][1]
        #   top_lip_lowest_y = shape[62][1]

          return (True,bottom_lip_highest_y, top_lip_lowest_y)