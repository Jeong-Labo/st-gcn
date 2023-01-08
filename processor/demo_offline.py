#!/usr/bin/env python
import os
import sys
import argparse
import json
import shutil
import time

import numpy as np
import torch
from torchvision import transforms
import skvideo.io

from .io import IO
import tools
import tools.utils as utils

import cv2

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint


class DemoOffline(IO):

    def start(self):
        # initiate
        label_name_path = './resource/kinetics_skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        # pose estimation
        video, data_numpy = self.pose_estimation()

        # action recognition
        data = torch.from_numpy(data_numpy)
        data = data.unsqueeze(0)
        data = data.float().to(self.dev).detach()  # (1, channel, frame, joint, person)

        # model predict
        voting_label_name, video_label_name, output, intensity = self.predict(data)

        # render the video
        images = self.render_video(data_numpy, voting_label_name,
                            video_label_name, intensity, video)

        # visualize
        fmt = cv2.VideoWriter_fourcc(*"mp4v")
        img_shape = next(images).shape
        writer = cv2.VideoWriter("./out.mp4", fmt, 30, (img_shape[1], img_shape[0]))
        for image in images:
            image = image.astype(np.uint8)
            #cv2.imshow("ST-GCN", image)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            writer.write(image)
        writer.release()

    def predict(self, data):
        # forward
        with torch.no_grad():
            output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature*feature).sum(dim=0)**0.5
        intensity = intensity.cpu().detach().numpy()

        # get result
        # classification result of the full sequence
        voting_label = output.sum(dim=3).sum(
            dim=2).sum(dim=1).argmax(dim=0)
        voting_label_name = self.label_name[voting_label]
        # classification result for each person of the latest frame
        num_person = data.size(4)
        latest_frame_label = [output[:, :, :, m].sum(
            dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]
        latest_frame_label_name = [self.label_name[l]
                                   for l in latest_frame_label]

        num_person = output.size(3)
        num_frame = output.size(1)
        video_label_name = list()
        for t in range(num_frame):
            frame_label_name = list()
            for m in range(num_person):
                person_label = output[:, t, :, m].sum(dim=1).argmax(dim=0)
                person_label_name = self.label_name[person_label]
                frame_label_name.append(person_label_name)
            video_label_name.append(frame_label_name)
        return voting_label_name, video_label_name, output, intensity

    def render_video(self, data_numpy, voting_label_name, video_label_name, intensity, video):
        images = utils.visualization.stgcn_visualize(
            data_numpy,
            self.model.graph.edge,
            intensity, video,
            voting_label_name,
            video_label_name,
            self.arg.height)
        return images

    def pose_estimation(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        weigths = torch.load('./weights/yolov7-w6-pose.pt')
        yolo = weigths['model']
        yolo = yolo.half().to(device)
        yolo.eval()

        # initiate
        self.model.eval()
        video_capture = cv2.VideoCapture(self.arg.video)
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_tracker = naive_pose_tracker(data_frame=video_length)

        # pose estimation
        start_time = time.time()
        frame_index = 0
        video = list()

        transform = transforms.ToTensor()
        while(True):
            # get image
            _, orig_image = video_capture.read()
            if orig_image is None:
                break
            image = letterbox(orig_image, 960, stride=64, auto=True)[0]
            source_H, source_W, _ = orig_image.shape
            orig_image = cv2.resize(
                orig_image, (256 * source_W // source_H, 256))
            H, W, _ = image.shape
            video.append(orig_image)

            image = transform(image)
            image = image.unsqueeze(0).to(device)
            with torch.no_grad():
                output, _ = yolo(image.half())
            output = non_max_suppression_kpt(output, 0.25, 0.65, nc=yolo.yaml['nc'], nkpt=yolo.yaml['nkpt'], kpt_label=True)
            with torch.no_grad():
                output = output_to_keypoint(output)
            if(len(output) == 0):
                output = output.reshape(0, 58)
            print(output.shape)
            multi_pose = [
                output[:, 7+3*0:10+3*0],
                (output[:, 7+3*6:10+3*6] + output[:, 7+3*5:10+3*5]) / 2,
                output[:, 7+3*6:10+3*6],
                output[:, 7+3*8:10+3*8],
                output[:, 7+3*10:10+3*10],
                output[:, 7+3*5:10+3*5],
                output[:, 7+3*7:10+3*7],
                output[:, 7+3*9:10+3*9],
                output[:, 7+3*12:10+3*12],
                output[:, 7+3*14:10+3*14],
                output[:, 7+3*16:10+3*16],
                output[:, 7+3*11:10+3*11],
                output[:, 7+3*13:10+3*13],
                output[:, 7+3*15:10+3*15],
                output[:, 7+3*2:10+3*2],
                output[:, 7+3*1:10+3*1],
                output[:, 7+3*4:10+3*4],
                output[:, 7+3*3:10+3*3],
            ]
            multi_pose = np.stack(multi_pose, axis=1)
            """
            nimg = image[0].permute(1, 2, 0) * 255
            nimg = nimg.cpu().numpy().astype(np.uint8)
            nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
            for idx in range(output.shape[0]):
                plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
            """

            # normalization
            multi_pose[:, :, 0] = multi_pose[:, :, 0] / W
            multi_pose[:, :, 1] = multi_pose[:, :, 1] / H
            multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
            multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
            multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

            # pose tracking
            pose_tracker.update(multi_pose, frame_index)
            frame_index += 1

            print('Pose estimation ({}/{}).'.format(frame_index, video_length))

        del yolo
        data_numpy = pose_tracker.get_skeleton_sequence()
        return video, data_numpy

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--video',
                            default='./resource/media/skateboarding.mp4',
                            help='Path to video')
        parser.add_argument('--openpose',
                            default=None,
                            help='Path to openpose')
        parser.add_argument('--model_input_frame',
                            default=128,
                            type=int)
        parser.add_argument('--model_fps',
                            default=30,
                            type=int)
        parser.add_argument('--height',
                            default=1080,
                            type=int,
                            help='height of frame in the output video.')
        parser.set_defaults(
            config='./config/st_gcn/kinetics-skeleton/demo_offline.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser

class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    """

    def __init__(self, data_frame=128, num_joint=18, max_frame_dis=np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info = list()

    def update(self, multi_pose, current_frame):
        # multi_pose.shape: (num_person, num_joint, 3)

        if current_frame <= self.latest_frame:
            return

        if len(multi_pose.shape) != 3:
            return

        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)
        for p in multi_pose[score_order]:

            # match existing traces
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                # trace.shape: (num_frame, num_joint, 3)
                if current_frame <= latest_frame:
                    continue
                mean_dis, is_close = self.get_dis(trace, p)
                if is_close:
                    if matching_trace is None:
                        matching_trace = trace_index
                        matching_dis = mean_dis
                    elif matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis

            # update trace information
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]

                # padding zero if the trace is fractured
                pad_mode = 'interp' if latest_frame == self.latest_frame else 'zero'
                pad = current_frame-latest_frame-1
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)

            else:
                new_trace = np.array([p])
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame

    def get_skeleton_sequence(self):

        # remove old traces
        valid_trace_index = []
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            if self.latest_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        self.trace_info = [self.trace_info[v] for v in valid_trace_index]

        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None

        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.latest_frame - latest_frame)
            d = trace[-end:]
            beg = end - len(d)
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))

        return data

    # concatenate pose to a trace
    def cat_pose(self, trace, pose, pad, pad_mode):
        # trace.shape: (num_frame, num_joint, 3)
        num_joint = pose.shape[0]
        num_channel = pose.shape[1]
        if pad != 0:
            if pad_mode == 'zero':
                trace = np.concatenate(
                    (trace, np.zeros((pad, num_joint, 3))), 0)
            elif pad_mode == 'interp':
                last_pose = trace[-1]
                coeff = [(p+1)/(pad+1) for p in range(pad)]
                interp_pose = [(1-c)*last_pose + c*pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    # calculate the distance between a existing trace and the input pose

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy)**2).sum(1))**0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close
