
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import comm
from utils.timer import Timer
from collections import defaultdict
from .visualize_infer import show_image_with_boxes
import cv2 as cv

def compute_on_dataset(model, data_loader, device, timer=None, vis=False,
                        eval_score_iou=False, eval_depth=False, eval_trunc_recall=False):
    
    model.eval()
    cpu_device = torch.device("cpu")

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):
            images, targets, image_ids = batch["images"], batch["targets"], batch["img_ids"]
            images = images.to(device)

            # extract label data for visualize
            vis_target = targets[0]
            targets = [target.to(device) for target in targets]

            if timer:
                timer.tic()

            output, eval_utils, visualize_preds = model(images, targets)
            output = output.to(cpu_device)

            if timer:
                torch.cuda.synchronize()
                timer.toc()


            if vis:
                img = show_image_with_boxes(vis_target.get_field('ori_img'), output, vis_target,
                                    visualize_preds, vis_scores=eval_utils['vis_scores'])
                img = img[:,:,(2,1,0)]

                cv.imshow("result",img)
                cv.waitKey(10)

                # plt.figure()
                # # plt.subplot(211)
                # # plt.imshow(all_heatmap); plt.title('heatmap'); plt.axis('off')
                # # plt.subplot(212)
                # plt.imshow(img);
                # plt.title('2D/3D boxes');
                # plt.axis('off')
                # # plt.suptitle('Detections')
                # plt.savefig("/home/hovexb/CODE/MonoFlex/output/exp/imgs/{}.png".format(idx))
                # plt.ion()
                # plt.pause(0.4)
                # plt.close()



def inference(
        model,
        data_loader,
        dataset_name,
        eval_types=("detections",),
        device="cuda",
        output_folder=None,
        metrics=['R40'],
        vis=False,
        eval_score_iou=False,
):
    device = torch.device(device)
    num_devices = comm.get_world_size()
    # logger = logging.getLogger("monoflex.inference")
    #dataset = data_loader.dataset
    # logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    # predict_folder = os.path.join(output_folder, 'data')
    # os.makedirs(predict_folder, exist_ok=True)

    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()

    compute_on_dataset(model, data_loader, device,
                      inference_timer, vis, eval_score_iou)
    comm.synchronize()




