import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from medpy import metric
import torch
from torchvision.transforms import ToTensor
from datetime import datetime

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_list(dir, begin=0, end=150):
    images = []

    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        idx = 0
        for fname in fnames:
            if is_image_file(fname):
                if idx >= begin and idx <= end:
                    path = os.path.join(root, fname)

                    subname = path.split("/")
                    images.append(os.path.join(subname[-2], subname[-1]))
                idx += 1
    print(len(images))
    return images


def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max(
        [
            (1 + beta_square) * p * r / (beta_square * p + r)
            for p, r in zip(precision, recall)
        ]
    )

    return max_fmeasure


def computeBER_mth(gt_path, pred_path, begin=0, end=150, exp_id=None):
    print(f"gt_path:{gt_path}")
    print(f"pred_path:{pred_path}")

    gt_list = get_image_list(gt_path, begin, end)
    nim = len(gt_list)

    stats = np.zeros((nim, 4), dtype="float")
    stats_jaccard = np.zeros(nim, dtype="float")
    stats_mae = np.zeros(nim, dtype="float")
    stats_fscore = np.zeros((256, nim, 2), dtype="float")
    curTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 创建一个文件来保存结果
    with open("/share/codes/lzc/myModels/shadowdiff/saves/metrics_output.txt", "a") as f:
        f.write(f"--------------{curTime}----------------\n")
        f.write(f"--------------{exp_id}----------------\n")
        f.write(f"gt_path:{gt_path}\n")
        f.write(f"pred_path:{pred_path}\n")

        for i in tqdm(range(0, len(gt_list)), desc="Calculating Metrics:"):
            im = gt_list[i]
            GTim = np.asarray(Image.open(os.path.join(gt_path, im)).convert("L"))
            posPoints = GTim > 0.5
            negPoints = GTim <= 0.5
            countPos = np.sum(posPoints.astype("uint8"))
            countNeg = np.sum(negPoints.astype("uint8"))
            sz = GTim.shape
            GTim = GTim > 0.5

            Predim = np.asarray(
                Image.open(os.path.join(pred_path, im))
                .convert("L")
                .resize((sz[1], sz[0]), Image.NEAREST)
            )

            # BER
            tp = (Predim > 102) & posPoints
            tn = (Predim <= 102) & negPoints
            countTP = np.sum(tp)
            countTN = np.sum(tn)
            stats[i, :] = [countTP, countTN, countPos, countNeg]

            # IoU
            pred_iou = Predim > 102
            stats_jaccard[i] = metric.binary.jc(pred_iou, posPoints)

            # MAE
            pred_mae = Predim > 12
            mae_value = np.mean(
                np.abs(pred_mae.astype(float) - posPoints.astype(float))
            )
            stats_mae[i] = mae_value

            # Precision and Recall for FMeasure
            eps = 1e-4
            for jj in range(0, 256):
                real_tp = np.sum((Predim > jj) & posPoints)
                real_t = countPos
                real_p = np.sum((Predim > jj).astype("uint8"))

                precision_value = (real_tp + eps) / (real_p + eps)
                recall_value = (real_tp + eps) / (real_t + eps)
                stats_fscore[jj, i, :] = [precision_value, recall_value]

        # Print BER
        posAcc = np.sum(stats[:, 0]) / np.sum(stats[:, 2])
        negAcc = np.sum(stats[:, 1]) / np.sum(stats[:, 3])
        pA = 100 - 100 * posAcc
        nA = 100 - 100 * negAcc
        BER = 0.5 * (2 - posAcc - negAcc) * 100
        print("BER, S-BER, N-BER:")
        print(BER, pA, nA)
        f.write("BER, S-BER, N-BER:\n")
        f.write(f"{BER}, {pA}, {nA}\n")

        # Print IoU
        jaccard_value = np.mean(stats_jaccard)
        print("IoU:", jaccard_value)
        f.write(f"IoU: {jaccard_value}\n")

        # Print MAE
        mean_mae_value = np.mean(stats_mae)
        print("MAE:", mean_mae_value)
        f.write(f"MAE: {mean_mae_value}\n")

        # Print Fmeasure
        precision_threshold_list = np.mean(stats_fscore[:, :, 0], axis=1).tolist()
        recall_threshold_list = np.mean(stats_fscore[:, :, 1], axis=1).tolist()
        fmeasure = cal_fmeasure(precision_threshold_list, recall_threshold_list)
        print("Fmeasure:", fmeasure)
        f.write(f"Fmeasure: {fmeasure}\n")

    return {
        "BER": BER,
        "S-BER": pA,
        "N-BER": nA,
        "IoU": jaccard_value,
        "MAE": mean_mae_value,
        "Fmeasure": fmeasure,
    }


def computeBER_mth_gpu(gt_path, pred_path, begin=0, end=150, exp_id=None):
    print(f"gt_path:{gt_path}")
    print(f"pred_path:{pred_path}")

    gt_list = get_image_list(gt_path, begin, end)
    nim = len(gt_list)

    stats = torch.zeros((nim, 4), dtype=torch.float32, device="cuda")
    stats_jaccard = torch.zeros(nim, dtype=torch.float32, device="cuda")
    stats_mae = torch.zeros(nim, dtype=torch.float32, device="cuda")
    stats_fscore = torch.zeros((256, nim, 2), dtype=torch.float32, device="cuda")
    curTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 创建一个文件来保存结果
    with open("/share/codes/lzc/myModels/shadowdiff/saves/metrics_output.txt", "a") as f:

        f.write(f"--------------{curTime}----------------\n")
        f.write(f"--------------{exp_id}----------------\n")
        f.write(f"gt_path:{gt_path}\n")
        f.write(f"pred_path:{pred_path}\n")

        for i in tqdm(range(0, len(gt_list)), desc="Calculating Metrics:"):
            im = gt_list[i]
            GTim = torch.tensor(
                np.asarray(Image.open(os.path.join(gt_path, im)).convert("L")),
                dtype=torch.float32,
                device="cuda",
            )
            posPoints = GTim > 0.5
            negPoints = GTim <= 0.5
            countPos = torch.sum(posPoints.to(torch.uint8))
            countNeg = torch.sum(negPoints.to(torch.uint8))
            sz = GTim.shape
            GTim = GTim > 0.5

            Predim = torch.tensor(
                np.asarray(
                    Image.open(os.path.join(pred_path, im))
                    .convert("L")
                    .resize((sz[1], sz[0]), Image.NEAREST)
                ),
                dtype=torch.float32,
                device="cuda",
            )

            # BER
            tp = (Predim > 102) & posPoints
            tn = (Predim <= 102) & negPoints
            countTP = torch.sum(tp)
            countTN = torch.sum(tn)
            stats[i, :] = torch.tensor(
                [countTP, countTN, countPos, countNeg], device="cuda"
            )

            # IoU
            pred_iou = Predim > 102
            stats_jaccard[i] = metric.binary.jc(
                pred_iou.cpu().numpy(), posPoints.cpu().numpy()
            )  # medpy does not support GPU

            # MAE
            pred_mae = Predim > 12
            mae_value = torch.mean(
                torch.abs(pred_mae.to(torch.float32) - posPoints.to(torch.float32))
            )
            stats_mae[i] = mae_value

            # Precision and Recall for FMeasure
            eps = 1e-4
            for jj in range(0, 256):
                real_tp = torch.sum((Predim > jj) & posPoints)
                real_t = countPos
                real_p = torch.sum((Predim > jj).to(torch.uint8))

                precision_value = (real_tp + eps) / (real_p + eps)
                recall_value = (real_tp + eps) / (real_t + eps)
                stats_fscore[jj, i, :] = torch.tensor(
                    [precision_value, recall_value], device="cuda"
                )

        # Print BER
        posAcc = torch.sum(stats[:, 0]) / torch.sum(stats[:, 2])
        negAcc = torch.sum(stats[:, 1]) / torch.sum(stats[:, 3])
        pA = 100 - 100 * posAcc
        nA = 100 - 100 * negAcc
        BER = 0.5 * (2 - posAcc - negAcc) * 100
        print("BER, S-BER, N-BER:")
        print(BER.item(), pA.item(), nA.item())
        f.write("BER, S-BER, N-BER:\n")
        f.write(f"{BER.item()}, {pA.item()}, {nA.item()}\n")

        # Print IoU
        jaccard_value = torch.mean(stats_jaccard)
        print("IoU:", jaccard_value.item())
        f.write(f"IoU: {jaccard_value.item()}\n")

        # Print MAE
        mean_mae_value = torch.mean(stats_mae)
        print("MAE:", mean_mae_value.item())
        f.write(f"MAE: {mean_mae_value.item()}\n")

        # Print Fmeasure
        precision_threshold_list = torch.mean(stats_fscore[:, :, 0], dim=1).tolist()
        recall_threshold_list = torch.mean(stats_fscore[:, :, 1], dim=1).tolist()
        fmeasure = cal_fmeasure(precision_threshold_list, recall_threshold_list)
        print("Fmeasure:", fmeasure)
        f.write(f"Fmeasure: {fmeasure}\n")

    return {
        "BER": BER.item(),
        "S-BER": pA.item(),
        "N-BER": nA.item(),
        "IoU": jaccard_value.item(),
        "MAE": mean_mae_value.item(),
        "Fmeasure": fmeasure,
    }


# computeBER_mth_fast = torch.compile(computeBER_mth,mode='max-autotune')


toTensor = ToTensor()


def Visha_scores(config: dict, begin=0, end=1000, gpu=False):
    # gt_path = os.path.join("/share", "datasets", "ViSha_release", "test", "labels")
    gt_path = config["visha"]["test_gt_root"]
    pred_path = config["eval_output_dir"]
    exp_id = config['exp_id']
    if gpu:
        computeBER_mth_gpu(gt_path, pred_path, begin, end, exp_id)
    else:
        computeBER_mth(gt_path, pred_path, begin, end, exp_id)
