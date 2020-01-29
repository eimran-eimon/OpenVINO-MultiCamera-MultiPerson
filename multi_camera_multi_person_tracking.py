"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import time
import queue
from threading import Thread
import json
import logging as log
import sys

import cv2 as cv

from utils.network_wrappers import Detector, VectorCNN
from mc_tracker.mct import MultiCameraTracker
from utils.misc import read_py_config
from utils.video import MulticamCapture
from utils.visualization import visualize_multicam_detections

from db.insert_coord import close_db_conn
from db.insert_coord import insert_data

'''
/open_model_zoo-master/tools/downloader/intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml


./interactive_face_detection_demo     -i ~/Downloads/nysm.mp4 -m /opt/intel/openvino_2019.3.376/deployment_tools/open_model_zoo/tools/downloader/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml     -m_ag ~/Downloads/open_model_zoo-master/tools/downloader/intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml    -m_hp /opt/intel/openvino_2019.3.376/deployment_tools/open_model_zoo/tools/downloader/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml     -m_em ~/Downloads/open_model_zoo-master/tools/downloader/intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml     -m_lm /opt/intel/openvino_2019.3.376/deployment_tools/open_model_zoo/tools/downloader/intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml


source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6
python multi_camera_multi_person_tracking.py \
    -i resources/Pedestrain_Detect_2_1_1.mp4 resources/Pedestrain_Detect_2_1_1.mp4 \
    --m_detector /opt/intel/openvino_2019.3.376/deployment_tools/open_model_zoo/tools/downloader/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml \
    --m_reid /opt/intel/openvino_2019.3.376/deployment_tools/open_model_zoo/tools/downloader/intel/person-reidentification-retail-0076/FP32/person-reidentification-retail-0076.xml \
    --config config.py \
    --cpu_extension ~/inference_engine_demos_build/intel64/Release/lib/libcpu_extension.so

python multi_camera_multi_person_tracking.py \
    -i resources/campus_1.avi resources/campus_2.avi  \
    --m_detector /opt/intel/openvino_2019.3.376/deployment_tools/open_model_zoo/tools/downloader/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml \
    --m_reid /opt/intel/openvino_2019.3.376/deployment_tools/open_model_zoo/tools/downloader/intel/person-reidentification-retail-0076/FP32/person-reidentification-retail-0076.xml \
    --config config.py \
    --cpu_extension ~/inference_engine_demos_build/intel64/Release/lib/libcpu_extension.so

source /opt/intel/openvino/bin/setupvars.sh \
python multi_camera_multi_person_tracking.py     -i 2     --m_detector /opt/intel/openvino_2019.3.376/deployment_tools/open_model_zoo/tools/downloader/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml     --m_reid /opt/intel/openvino_2019.3.376/deployment_tools/open_model_zoo/tools/downloader/intel/person-reidentification-retail-0076/FP32/person-reidentification-retail-0076.xml     --config config.py     --cpu_extension ~/inference_engine_demos_build/intel64/Release/lib/libcpu_extension.so --history_file ./history.txt --t_detector 0.5



'''

log.basicConfig(stream=sys.stdout, level=log.DEBUG)


class FramesThreadBody:
    def __init__(self, capture, max_queue_length=2):
        self.process = True
        self.frames_queue = queue.Queue()
        self.capture = capture
        self.max_queue_length = max_queue_length

    def __call__(self):
        while self.process:
            if self.frames_queue.qsize() > self.max_queue_length:
                time.sleep(0.1)
            has_frames, frames = self.capture.get_frames()
            if not has_frames and self.frames_queue.empty():
                self.process = False
                break
            if has_frames:
                self.frames_queue.put(frames)


def run(params, capture, detector, reid):
    win_name = 'CCTV Person tracking'
    frame_no = 0
    config = {}
    if len(params.config):
        config = read_py_config(params.config)

    tracker = MultiCameraTracker(capture.get_num_sources(), reid, **config)

    thread_body = FramesThreadBody(capture, max_queue_length=len(capture.captures) * 2)
    frames_thread = Thread(target=thread_body)
    frames_thread.start()

    if len(params.output_video):
        video_output_size = (1920 // capture.get_num_sources(), 1080)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        output_video = cv.VideoWriter(params.output_video,
                                      fourcc, 24.0,
                                      video_output_size)
    else:
        output_video = None

    while cv.waitKey(1) != 27 and thread_body.process:

        start = time.time()
        try:
            frames = thread_body.frames_queue.get_nowait()
        except queue.Empty:
            frames = None

        if frames is None:
            continue
        if (frame_no % 2) == 0:
            # print(frame_no)
            all_detections = detector.get_detections(frames)
            all_masks = [[] for _ in range(len(all_detections))]
            for i, detections in enumerate(all_detections):
                all_detections[i] = [det[0] for det in detections]
                all_masks[i] = [det[2] for det in detections if len(det) == 3]
            tracker.process(frames, all_detections, all_masks)

        tracked_objects = tracker.get_tracked_objects()
        fps = round(1 / (time.time() - start), 1)
        vis = visualize_multicam_detections(frames, tracked_objects, fps)
        insert_data(frames, tracked_objects)
        # cv.namedWindow(win_name, cv.WINDOW_NORMAL)
        cv.line(vis, (256, 351), (481, 353), (0, 0, 255), 3)
        cv.imshow(win_name, vis)
        cv.setMouseCallback(win_name,click_and_crop)
        frame_no = frame_no + 1
        if output_video:
            output_video.write(cv.resize(vis, video_output_size))

    thread_body.process = False
    frames_thread.join()

    if len(params.history_file):
        history = tracker.get_all_tracks_history()
        with open(params.history_file, 'w') as outfile:
            json.dump(history, outfile)


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        print(*refPt)

def main():
    """Prepares data for the person recognition demo"""
    parser = argparse.ArgumentParser(description='Multi camera multi person \
                                                  tracking live demo script')
    parser.add_argument('-i', type=str, nargs='+', help='Input sources (indexes \
                        of cameras or paths to video files)', required=True)

    parser.add_argument('-m', '--m_detector', type=str, required=True,
                        help='Path to the person detection model')
    parser.add_argument('--t_detector', type=float, default=0.6,
                        help='Threshold for the person detection model')

    parser.add_argument('--m_reid', type=str, required=True,
                        help='Path to the person reidentification model')

    parser.add_argument('--output_video', type=str, default='', required=False)
    parser.add_argument('--config', type=str, default='', required=False)
    parser.add_argument('--history_file', type=str, default='', required=False)

    parser.add_argument('-d', '--device', type=str, default='CPU')
    parser.add_argument('-l', '--cpu_extension',
                        help='MKLDNN (CPU)-targeted custom layers. Absolute \
                              path to a shared library with the kernels impl.',
                        type=str, default=None)

    args = parser.parse_args()

    capture = MulticamCapture(args.i)

    person_detector = Detector(args.m_detector, args.t_detector,
                               args.device, args.cpu_extension,
                               capture.get_num_sources())

    if args.m_reid:
        person_recognizer = VectorCNN(args.m_reid, args.device)
    else:
        person_recognizer = None
    run(args, capture, person_detector, person_recognizer)
    close_db_conn()
    log.info('Process finished successfully')


if __name__ == '__main__':
    main()
