from torch.utils.data import Dataset
import os
from PIL import Image

class GbufferDataset(Dataset):
    def __init__(self, noisy_img_path, ground_truth_path, transform=None, views=30) -> None:
        super().__init__()
        self.transform = transform
    

    def handle_dataset(self, noisy_img_path, ground_truth_path, views):
        stepsize = 360.0 / views
        rotations = ['{0:03d}'.format(rotation) for rotation in range(0, 360, int(stepsize))]
        model_ids = sorted(os.listdir(noisy_img_path))
        noisy_imgs = []
        Galbedos = []
        Gdepths = []
        Gnormals = []
        ground_truths = []
        for model_id in model_ids:
            for rotation in rotations:
                noisy_img = os.path.join(noisy_img_path, model_id, '{}_r_{}.png'.format(model_id, rotation))
                Galbedo = os.path.join(noisy_img_path, model_id, '{}_r_{}_albedo0001.png'.format(model_id, rotation))
                Gdepth = os.path.join(noisy_img_path, model_id, '{}_r_{}_depth0001.png'.format(model_id, rotation))
                Gnormal = os.path.join(noisy_img_path, model_id, '{}_r_{}_normal0001.png'.format(model_id, rotation))
                ground_truth = os.path.join(ground_truth_path, model_id, '{}_r_{}.png'.format(model_id, rotation))
                noisy_imgs.append(noisy_img)
                Galbedos.append(Galbedo)
                Gdepths.append(Gdepth)
                Gnormals.append(Gnormal)
                ground_truths.append(ground_truth)
        return noisy_imgs, Galbedos, Gdepths, Gnormals, ground_truths