import os
def get_root_dir():
    return os.path.dirname(os.path.dirname(__file__))
import sys
sys.path.append(get_root_dir())
import cv2
import numpy as np
import torch
import argparse
import yaml
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text
import glob
import os
from PIL import Image, ImageDraw, ImageFont
import math

class SemSeg:
    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.labels = eval(cfg['DATASET']['NAME']).CLASSES

        # initialize the model and load weights and send to device
        dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
        # self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette), cfg['DATASET']['MODALS'])
        self.model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], len(self.palette), dataset_cfg['MODALS'])
        msg = self.model.load_state_dict(torch.load(cfg['EVAL']['MODEL_PATH'], map_location='cpu'))
        print(msg)
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline_img = T.Compose([
            # T.Resize(self.size),
            T.Lambda(lambda x: x / 255),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])
        self.tf_pipeline_modal = T.Compose([
            # T.Resize(self.size),
            T.Lambda(lambda x: x / 255),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        seg_image = self.palette[seg_map].squeeze()
        if overlay:
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        image = seg_image.to(torch.uint8)
        pil_image = Image.fromarray(image.numpy())
        return [pil_image, seg_map]
        # return seg_map

    @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)

    def _open_img(self, file):
        # img = io.read_image(file)
        img = torch.tensor(cv2.imread(file)[:, :, (2, 1, 0)].transpose(2, 0, 1))
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def predict(self, img_fname: str, overlay: bool) -> Tensor:
        if cfg['DATASET']['NAME'] == 'DELIVER':
            x1 = img_fname.replace('/img', '/hha').replace('_rgb', '_depth')
            x2 = img_fname.replace('/img', '/lidar').replace('_rgb', '_lidar')
            x3 = img_fname.replace('/img', '/event').replace('_rgb', '_event')
            lbl_path = img_fname.replace('/img', '/semantic').replace('_rgb', '_semantic')
        elif cfg['DATASET']['NAME'] == 'KITTI360':
            x1 = os.path.join(img_fname.replace('data_2d_raw', 'data_2d_hha'))
            x2 = os.path.join(img_fname.replace('data_2d_raw', 'data_2d_lidar'))
            x2 = x2.replace('.png', '_color.png')
            x3 = os.path.join(img_fname.replace('data_2d_raw', 'data_2d_event'))
            x3 = x3.replace('/image_00/data_rect/', '/').replace('.png', '_event_image.png')
            lbl_path = os.path.join(
                *[img_fname.replace('data_2d_raw', 'data_2d_semantics/train').replace('data_rect', 'semantic')])
        elif cfg['DATASET']['NAME'] == 'Mydata':
            x1 = img_fname.replace('\\img', '\\building_height')
            x2 = img_fname.replace('\\img', '\\osm_building')
            x3 = img_fname.replace('\\img', '\\ndvi')
            lbl_path = img_fname.replace('\\img', '\\semantic')
        elif cfg['DATASET']['NAME'] == 'Mydata_adapt':
            x1 = img_fname.replace('\\img', '\\building_height')
            x2 = img_fname.replace('\\img', '\\osm_building')
            x3 = img_fname.replace('\\img', '\\ndvi')


        # image = io.read_image(img_fname)[:3, ...]
        ndvi = self._open_img(x3)
        img = cv2.imread(img_fname)
        image = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(1,2).transpose(0,1)

        img = self.tf_pipeline_img(image).to(self.device)
        # --- modals
        x1 = self._open_img(x1)
        x1 = self.tf_pipeline_modal(x1).to(self.device)
        x2 = self._open_img(x2)
        x2 = self.tf_pipeline_modal(x2).to(self.device)
        x3 = self._open_img(x3)
        x3 = self.tf_pipeline_modal(x3).to(self.device)

        sample = [img, x1, x2, x3][:len(modals)]
        h, w = x1.shape[2:]
        num_classes = len(self.palette)
        new_pre0 = np.zeros((1, num_classes, h, w))
        repeat_ = 1-0.9  # overlay ratio 0.9
        for i in range(math.ceil(h / (self.size[0] * repeat_)) + 1):
            for j in range(math.ceil(w / (self.size[1] * repeat_)) + 1):
                h_s = math.ceil(i * self.size[0] * repeat_)
                w_s = math.ceil(j * self.size[1] * repeat_)
                h_s_ = math.ceil((i - 1) * self.size[0] * repeat_)
                w_s_ = math.ceil((j - 1) * self.size[1] * repeat_)
                if h_s_ + self.size[0] <= h and w_s_ + self.size[1] <= w:
                    rgb_ = image[:, h_s: h_s + self.size[0], w_s: w_s + self.size[1]].unsqueeze(0).detach().numpy()
                    x4_ = ndvi[:, h_s: h_s + self.size[0], w_s: w_s + self.size[1]].unsqueeze(0).detach().numpy()

                    new_sample = []
                    for data in sample:
                        data_ = data[:, :, h_s:h_s + self.size[0], w_s:w_s + self.size[1]]
                        data_h = data_.size(2)
                        data_w = data_.size(3)

                        if data_h < self.size[0] or data_w < self.size[1]:
                            new_data = torch.zeros((1, data_.size(1), self.size[0], self.size[1])).to(self.device)
                            new_data[:, :, 0:data_h, 0:data_w] = data_
                            data_ = new_data
                        new_sample.append(data_)
                    new_rgb = np.zeros((1, data_.size(1), self.size[0], self.size[1]))
                    new_rgb[:, :, 0:data_h, 0:data_w] = rgb_
                    new_x4 = np.zeros((1, data_.size(1), self.size[0], self.size[1]))
                    new_x4[:, :, 0:data_h, 0:data_w] = x4_
                    if 0 in new_sample[-1].unique() and len(new_sample[-1].unique())==1:
                        continue
                    else:
                        out = self.model_forward(new_sample)
                        seg_map0 = out
                        logits_seg_map0 = seg_map0.cpu().detach().numpy()

                        mask1 = (new_rgb[:, 0, :, :] == 0) & (new_rgb[:, 1, :, :] == 0) & (new_rgb[:, 2, :, :] == 0)
                        mask2 = (new_x4[:, 0, :, :] == 0) & (new_x4[:, 1, :, :] == 0) & (new_x4[:, 2, :, :] == 0)

                        mask = mask1 & mask2
                        mask = np.expand_dims(mask,axis=1)
                        mask = np.repeat(mask,2,axis=1)
                        if True in mask :
                            logits_seg_map0[mask] = 0

                    new_pre0[:, :, h_s:h_s + self.size[0], w_s:w_s + self.size[1]] += logits_seg_map0[:, :, 0:data_h,
                                                                                      0:data_w]

        seg_map = np.argmax(new_pre0, axis=1)
        seg_map = seg_map[0, :, :]
        seg_image = self.palette[seg_map].squeeze()
        image = seg_image.to(torch.uint8)
        pil_image = Image.fromarray(image.numpy())
        return [pil_image, seg_map]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='../configs/rv_UDASS.yaml') #../configs/rv_UDASS.yaml #../configs/rv_source_domain.yaml
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # cases = ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres', None]
    cases = ['lidarjitter']

    modals = cfg['DATASET']['MODALS']

    test_file = Path(cfg['TEST']['FILE'])
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    # print(f"Model {cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}")
    # print(f"Model {cfg['DATASET']['NAME']}")

    modals_name = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    save_dir = Path(cfg['SAVE_DIR']) / 'test_results' / (
                cfg['DATASET']['NAME'] + '_' + cfg['MODEL']['BACKBONE'] + '_' + modals_name)

    semseg = SemSeg(cfg)

    if test_file.is_file():
        segmap, logits_seg_map = semseg.predict(str(test_file), cfg['TEST']['OVERLAY'])
        segmap.save(save_dir / f"{str(test_file.stem)}.png")
    else:
        if cfg['DATASET']['NAME'] == 'DELIVER':
            files = sorted(glob.glob(os.path.join(*[str(test_file), 'img', '*', 'val', '*', '*.png'])))  # --- Deliver
        elif cfg['DATASET']['NAME'] == 'KITTI360':
            source = os.path.join(test_file, 'val.txt')
            files = []
            with open(source) as f:
                files_ = f.readlines()
            for item in files_:
                file_name = item.strip()
                if ' ' in file_name:
                    # --- KITTI-360
                    file_name = os.path.join(*[str(test_file), file_name.split(' ')[0]])
                files.append(file_name)
        elif cfg['DATASET']['NAME'] == 'Mydata_ChinaGR':
            files = sorted(glob.glob(os.path.join(*[str(test_file), 'img', 'test', '*.png'])))
        elif cfg['DATASET']['NAME'] == 'Mydata':
            files = sorted(glob.glob(os.path.join(*[str(test_file), 'img', 'test', '*.png'])))
        elif cfg['DATASET']['NAME'] == 'Mydata_adapt':
            files = sorted(glob.glob(os.path.join(*[str(test_file), 'img', 'test', '*.png'])))

        else:
            raise NotImplementedError()

        for file in files:
            print(file)
            file_name = os.path.basename(file)

            segmap = semseg.predict(file, cfg['TEST']['OVERLAY'])
            save_path1 = 'E:/open_code/TransRVNet/' + str(save_dir)  # os.path.join(str(save_dir),file_name)
            os.makedirs(save_path1, exist_ok=True)
            save_path0 = 'E:/open_code/TransRVNet/' + str(
                save_dir) + '_color'  # os.path.join(str(save_dir),file_name)
            os.makedirs(save_path0, exist_ok=True)
            segmap[0].save(save_path0 + '/' + file_name)
            cv2.imwrite(save_path1 + '/' + file_name, segmap[1])  # .cpu().detach().numpy())
