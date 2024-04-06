import torch
import torchvision
from lightning.pytorch.callbacks import Callback

class DisplayReults(Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            img, mask = batch
            img = img[:3]
            mask = mask[:3]
            outputs = outputs[:3]

            # create image grid
            img_grid = torchvision.utils.make_grid(img)
            mask_grid = torchvision.utils.make_grid(mask)
            res_grid = torchvision.utils.make_grid(outputs)

            # log to tensorboard
            trainer.logger.experiment.add_image("Validation/Image", img_grid, global_step=trainer.global_step)
            trainer.logger.experiment.add_image("Validation/Mask", mask_grid, global_step=trainer.global_step)
            trainer.logger.experiment.add_image("Validation/Result", res_grid, global_step=trainer.global_step)

