import os
import sys
import time
from tqdm import tqdm
from PIL import Image
import json
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from toolbox import get_dataset
from toolbox import averageMeter, runningScore
from toolbox import class_to_RGB, load_ckpt


def evaluate(logdir, save_predict=False):

    filename = '/result'
    outputfile = open(logdir + filename + '.txt', 'w')
    sys.stdout = outputfile

    cfg = None
    for file in os.listdir(logdir):
        if file.endswith('.json'):
            with open(os.path.join(logdir, file), 'r') as fp:
                cfg = json.load(fp)
    assert cfg is not None

    device = torch.device('cuda:0')
    testset = get_dataset(cfg)[-1]
    test_loader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=cfg['num_workers'])
    model = DEMMNet(num_classes=cfg['n_classes']).cuda()
    model = load_ckpt(logdir, model, kind='best', strict=True)

    running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    time_meter = averageMeter()

    save_path = os.path.join(logdir, 'predicts')
    if not os.path.exists(save_path) and save_predict:
        os.mkdir(save_path)

    with torch.no_grad():
        model.eval()
        for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
            time_start = time.time()
            depth = sample['depth'].to(device)
            image = sample['image'].to(device)
            label = sample['label'].to(device)

            predict = model(image, depth)[0]
            predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
            time_meter.update(time.time() - time_start, n=image.size(0))
            label = label.cpu().numpy()
            running_metrics_val.update(label, predict)

            if save_predict:
                predict = predict.squeeze(0)  # [1, h, w] -> [h, w]
                predict = class_to_RGB(predict, N=len(testset.cmap), cmap=testset.cmap)
                predict = Image.fromarray(predict)
                predict.save(os.path.join(save_path, sample['label_path'][0]))

    metrics = running_metrics_val.get_scores()
    print(metrics)

    outputfile.close()


if __name__ == '__main__':
    import argparse
    from toolbox.models.DEMMNet import DEMMNet
    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--logdir", type=str, help="run logdir", default="run/..")
    parser.add_argument("-s", type=bool, default=False, help="save predict or not")
    args = parser.parse_args()
    evaluate(args.logdir, save_predict=args.s)
