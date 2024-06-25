import time
import datetime
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean
from config import *
from misc import *
from CDI_NSTSEG import CDI_NSTSEG
from utils import builder, configurator, io, misc, ops, pipeline, recorder
import argparse
import json
from tqdm import tqdm
import shutil


def parse_config():
    parser = argparse.ArgumentParser("Training and evaluation script")
    parser.add_argument("--config", default="./configs/CDI_NSTSEG/CDI_NSTSEG.py", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--info", type=str)
    args = parser.parse_args()

    config = configurator.Configurator.fromfile(args.config)
    

    config.use_ddp = False
    if args.model_name is not None:
        config.model_name = args.model_name
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.info is not None:
        config.experiment_tag = args.info
    if args.load_from is not None:
        config.load_from = args.load_from

    with open(args.datasets_info, encoding="utf-8", mode="r") as f:
        datasets_info = json.load(f)

    tr_paths = {}
    for tr_dataset in config.datasets.train.path:
        if tr_dataset not in datasets_info:
            raise KeyError(f"{tr_dataset} not in {args.datasets_info}!!!")
        tr_paths[tr_dataset] = datasets_info[tr_dataset]
    config.datasets.train.path = tr_paths

    te_paths = {}
    for te_dataset in config.datasets.test.path:
        if te_dataset not in datasets_info:
            raise KeyError(f"{te_dataset} not in {args.datasets_info}!!!")
        te_paths[te_dataset] = datasets_info[te_dataset]
    config.datasets.test.path = te_paths

    config.proj_root = os.path.dirname(os.path.abspath(__file__))
    config.exp_name = misc.construct_exp_name(model_name=config.model_name, cfg=config)
    if args.resume_from is not None:
        config.resume_from = args.resume_from
        resume_proj_root = os.sep.join(args.resume_from.split("/")[:-3])
        if resume_proj_root.startswith("./"):
            resume_proj_root = resume_proj_root[2:]
        config.output_dir = os.path.join(config.proj_root, resume_proj_root)
        config.exp_name = args.resume_from.split("/")[-3]
    else:
        config.output_dir = os.path.join(config.proj_root, "output")
    config.path = misc.construct_path(output_dir=config.output_dir, exp_name=config.exp_name)
    return config


torch.manual_seed(2021)
device_ids = [0]
torch.cuda.set_device(device_ids[0])

results_path = './results'
check_mkdir(results_path)
exp_name = 'CDI_NSTSEG'
args = {
    'scale': 416,
    'save_results': True
}

print(torch.__version__)

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

results = OrderedDict()



def test_once(
    model, data_loader, save_path, tta_setting, clip_range=None, show_bar=False, desc="[TE]", to_minmax=False
):
    model.eval()
    model.is_training = False
    cal_total_seg_metrics = recorder.CalTotalMetric()

    pgr_bar = enumerate(data_loader)
    if show_bar:
        pgr_bar = tqdm(pgr_bar, total=len(data_loader), ncols=79, desc=desc)
    for batch_id, batch in pgr_bar:
        
        data=batch["data"]
        inputs_1 = data["image1.5"]
        inputs = data["image1.0"]
        inputs_0 = data["image0.5"]
        
        #labels = data["mask"]
        inputs = Variable(inputs).cuda(device_ids[0])
        
        inputs_1 = Variable(inputs_1).cuda(device_ids[0])
        
        inputs_0 = Variable(inputs_0).cuda(device_ids[0])
        
        
        _, _, _, prediction = model(inputs, inputs_1, inputs_0)
        
        
        probs = prediction.data.squeeze(0).cpu()
        
                
                

        for i, pred in enumerate(probs):
            mask_path = batch["info"]["mask_path"][i]
            mask_array = io.read_gray_array(mask_path, dtype=np.uint8)
            mask_h, mask_w = mask_array.shape
            pred = np.array(transforms.Resize((mask_h, mask_w))(to_pil(pred)))
            pred_l = Image.fromarray(pred).convert('L')

            if save_path:  
                save_name=os.path.basename(mask_path)
                pred_l.save(os.path.join(save_path, save_name))
                
            cal_total_seg_metrics.step(pred, mask_array, mask_path)
    fixed_seg_results = cal_total_seg_metrics.get_results()
    return fixed_seg_results



def testing(model, cfg):
    for data_name, data_path, loader in pipeline.get_te_loader(cfg):
        pred_save_path = os.path.join(cfg.path.save, data_name)
        os.mkdir(pred_save_path)
        cfg.te_logger.record(f"Results will be saved into {pred_save_path}")
        seg_results = test_once(
            model=model,
            save_path=pred_save_path,
            data_loader=loader,
            tta_setting=cfg.test.tta,
            clip_range=cfg.test.clip_range,
            show_bar=cfg.test.get("show_bar", False),
            to_minmax=cfg.test.get("to_minmax", False),
        )
        cfg.te_logger.record(f"Results on the testset({data_name}): {misc.mapping_to_str(data_path)}\n{seg_results}")
       


def main():
    cfg = parse_config()
    if not cfg.resume_from:
        misc.pre_mkdir(path_config=cfg.path)
        with open(cfg.path.cfg_copy, encoding="utf-8", mode="w") as f:
            f.write(cfg.pretty_text)
        shutil.copy(__file__, cfg.path.trainer_copy)

    cfg.tr_logger = recorder.TxtLogger(cfg.path.tr_log)
    cfg.te_logger = recorder.TxtLogger(cfg.path.te_log)

    net = CDI_NSTSEG(backbone_path).cuda(device_ids[0])

    net.load_state_dict(torch.load('/CDI_NSTSEG/best_model.pth'), False)
    print('Load {} succeed!'.format('CDI_NSTSEG.pth'))

    net.eval()
    with torch.no_grad():

      testing(model=net, cfg=cfg)

if __name__ == '__main__':
    main()
