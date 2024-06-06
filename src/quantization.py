import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
import torch.ao.quantization.quantize_fx as quantize_fx
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from tinynn.graph.quantization.quantizer import QATQuantizer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    DisplayReults,
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def calibration(model, dataloader, num_iterations):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    count = 0

    with torch.no_grad():
        for data in dataloader:
            img, _ = data
            img = img.to(device)
            model(img)

            count += 1
            if count >= num_iterations:
                break

    return model


@task_wrapper
def quantization(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model_class = hydra.utils.get_class(cfg.model._target_)
    model: LightningModule = model_class.load_from_checkpoint(cfg.ckpt_path)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    callbacks.append(DisplayReults())

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule)

    torch.save(model.net.state_dict(), cfg.save_path)

    if cfg.fuse_batch:
        log.info("Fuse modules!")
        model.net = quantize_fx.fuse_fx(model.net.eval())
        trainer.test(model=model, datamodule=datamodule)
        torch.save(model.net.state_dict(), cfg.save_path)

    if cfg.ptq or cfg.qat:
        quantizer = QATQuantizer(
            model.net,
            torch.randn(1, 3, 52, 52),
            work_dir=cfg.quantizer.work_dir,
            config=cfg.quantizer,
        )
        model.net = quantizer.quantize()

        if cfg.ptq:
            log.info("Post training quantization!")
            model.net.apply(torch.quantization.disable_fake_quant)
            model.net.apply(torch.quantization.enable_observer)

            calibration(model.net, datamodule.train_dataloader(), 50)

            model.net.apply(torch.quantization.disable_observer)
            model.net.apply(torch.quantization.enable_fake_quant)

        if cfg.qat:
            log.info("Quantization awareness training!")
            trainer.fit(model=model, datamodule=datamodule)

        with torch.no_grad():
            model.net.eval()
            model.net.cpu()
            quantized_model = torch.quantization.convert(model.net)
            torch.save(quantized_model.state_dict(), cfg.save_path)

        model.net = quantized_model

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        cfg.trainer["accelerator"] = "cpu"
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

        trainer.test(model=model, datamodule=datamodule)

    return None, None


@hydra.main(version_base="1.3", config_path="../configs", config_name="quantization.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    quantization(cfg)


if __name__ == "__main__":
    main()
