import cv2 as cv

import imutils

import numpy as np

import config
import datetime

import logging
from logging.handlers import TimedRotatingFileHandler

import re
import os


def visualize(input, faces, aims, video_time, thickness=2):
    if faces[1] is not None:
        for indx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            if indx in aims:
                cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 0, 255), thickness)
                y = coords[1] - 15 if coords[1] - 15 > 15 else coords[1] + 15
                cv.putText(input, "aim", (coords[0], y), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            else:
                cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (255, 255, 255), thickness)

    cv.putText(input, video_time, (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def parse_filename(file_path): # format: cam1-2021-10-15_20-41-10.mp4
    file_name = os.path.basename(file_path)
    arr = re.split("-|_|\.", file_name)
    cam = arr[0]
    date_time = [int(i) for i in arr[1:-1] if i.isdigit()]

    return cam, date_time


def log_init(logger_name):	
    logger_file_name = config.log_path + f"/{logger_name}.log.txt"
    logger = logging.getLogger(logger_name)

    fileHandler = TimedRotatingFileHandler(logger_file_name, when='midnight', interval=1, backupCount=7, encoding = "UTF-8")

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    # fileHandler = logging.FileHandler(logger_file_name, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)


def detect_face(img, detector, tm):
    scale = 1
    while scale > 0.1:
        img_width = int(img.shape[1] * scale)
        img_height = int(img.shape[0] * scale)
        img = cv.resize(img, (img_width, img_height))
        tm.start()

        detector.setInputSize((img_width, img_height))
        img_face = detector.detect(img)

        tm.stop()

        scale -= 0.1
        if img_face[1] is not None and len(img_face[1]) == 1:
            break

    if config.visualize_flag:
        visualize(img, img_face, [], '')
        cv.imshow("image", img)
        cv.waitKey(0)

    return img_face


def get_img_info(face_paths, detector, trash_logger, tm):
    img_info = []
    for face_path in face_paths:
        img = cv.imread(cv.samples.findFile(face_path))
        img = imutils.resize(img, width=500)

        img_face = detect_face(img, detector, tm)
        face_name = os.path.basename(face_path)

        if img_face[1] is None:
            trash_logger.error('Cannot find a face in {}'.format(face_name))
            continue

        if len(img_face[1]) != 1:
            trash_logger.error('More than one person has been detected in {}'.format(face_name))
            continue

        img_info.append(
            {
                'img': img,
                'img_face': img_face,
                'face_name': face_name,
            }
        )

        trash_logger.info('The face in {} has been successfully detected'.format(face_name))

    return img_info


def start_recognition(face_paths, video_paths):
    try:
        log_init("main_recognition")
        log_init("trash_recognition")
        trash_logger = logging.getLogger('trash_recognition')
        main_logger = logging.getLogger('main_recognition')

        detector = cv.FaceDetectorYN.create(
            config.models.get('face_detection'),
            "",
            (320, 320),
            config.detector_params.get('score_threshold'),
            config.detector_params.get('nms_threshold'),
            config.detector_params.get('top_k'),
        )

        tm = cv.TickMeter()
        face_recognition_model = config.models.get('face_recognition')
        recognizer = cv.FaceRecognizerSF.create(face_recognition_model, "")

        img_info = get_img_info(face_paths, detector, trash_logger, tm)

        if len(img_info) == 0:
            trash_logger.error('Faces are not identified in {}'.format(*face_paths))
            return

        for video in video_paths:
            video_name = os.path.basename(video)
            cam, date_time = parse_filename(video)

            if len(date_time) != 6:
                trash_logger.error(f"Invalid filename {video_name}")
                continue

            trash_logger.debug('Start recognition in {}'.format(video_name))
            cap = cv.VideoCapture(video)
            frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) * 1.0)
            frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) * 1.0)
            detector.setInputSize([frame_width, frame_height])

            fps = int(cap.get(cv.CAP_PROP_FPS))
            if fps == 0:
                trash_logger.error(f"Empty video {video_name}")
                continue
            try:
                start_time = datetime.datetime(*date_time)
            except Exception as err:
                trash_logger.error(f"Invalid filename {video_name}\n" + str(err))
                continue

            frame_indx = 0
            detected_key = False
            detected_indx = 0
            detected_count = 0
            recognized_count = 0
            while cv.waitKey(1) < 0:

                has_frame, frame = cap.read()

                frame_indx += 1
                detected_indx += 1
                seconds = frame_indx / fps
                seconds = float('{:.2f}'.format(seconds))
                if int(seconds) != seconds:
                    video_time = str(start_time + datetime.timedelta(seconds=seconds))[:-4]
                else:
                    video_time = str(start_time + datetime.timedelta(seconds=seconds))

                if not has_frame:
                    break

                frame = cv.resize(frame, (frame_width, frame_height))

                tm.start()
                frame_faces = detector.detect(frame)
                tm.stop()

                aims = []
                if frame_faces[1] is not None:
                    if not detected_key:
                        detected_key = True
                        detected_count += 1
                    elif detected_indx >= fps:
                        detected_count += 1
                        detected_indx = 0
                        detected_key = False
                        trash_logger.debug(f"Detected face: {detected_count}; recognized face: {recognized_count}; time: {video_time}; camera: {cam}")
                        detected_count = 0
                        recognized_count = 0
                    else:
                        detected_count += 1

                    cosine_similarity_threshold = config.recognition_params.get('cosine_similarity_threshold')
                    l2_similarity_threshold = config.recognition_params.get('l2_similarity_threshold')

                    for img in img_info:
                        img_face_align = recognizer.alignCrop(img['img'], img['img_face'][1][0])
                        img_face_feature = recognizer.feature(img_face_align)
                        face_name = img['face_name']

                        for indx, frame_face in enumerate(frame_faces[1]):

                            frame_faces_align = recognizer.alignCrop(frame, frame_face)

                            frame_faces_feature = recognizer.feature(frame_faces_align)

                            cosine_score = recognizer.match(frame_faces_feature, img_face_feature, cv.FaceRecognizerSF_FR_COSINE)
                            cosine_score = float("{0:.3f}".format(cosine_score))
                            l2_score = recognizer.match(frame_faces_feature, img_face_feature, cv.FaceRecognizerSF_FR_NORM_L2)
                            l2_score = float("{0:.3f}".format(l2_score))

                            if cosine_score >= cosine_similarity_threshold or l2_score < l2_similarity_threshold:
                                recognized_count += 1
                                aims.append(indx)
                                trash_logger.info(f'FASE RECOGNIZED {face_name}. Cosine Similarity: {cosine_score}; NormL2 Distance: {l2_score}; Video time: {video_time}; camera: {cam}')
                                main_logger.info(f'Recognized face {face_name} at {video_time} on camera {cam}')

                if config.visualize_flag:
                    visualize(frame, frame_faces, aims, video_time)
                    cv.imshow('Live', frame)

            trash_logger.debug('Completed recognition in {}\n'.format(video_name))


        if config.visualize_flag:
            cv.destroyAllWindows()
    except Exception as err:
        main_logger.error(str(err))