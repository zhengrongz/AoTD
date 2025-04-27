import torch
import cv2
import numpy as np
from torchvision.transforms import Resize

class Normalize(object):

    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = torch.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


class Preprocessing(object):

    def __init__(self):
        self.norm = Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])

    def __call__(self, video):
        resized_video = []
        torch_resize = Resize([224,224])
        resized_video = video.permute(0,3,1,2)
        resized_video = torch_resize(resized_video)
        # for frame in video:
        #     #frame = frame.asnumpy()
        #     #resized_frame = cv2.resize(frame, (224, 224))
        #     resized_frame = torch_resize(frame.permute(2,0,1))
        #     resized_video.append(resized_frame)
        # resized_video = np.stack(resized_video, axis=0)
        resized_video = resized_video / 255.0
        #resized_video = torch.tensor(resized_video)
        #resized_video = resized_video.permute(0,3,1,2)
        resized_video = self.norm(resized_video)

        return resized_video
