from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS


@HOOKS.register_module()
class YOLOXMTLModeSwitchHook(Hook):
    """YOLOX mode-switch hook compatible with both MultiImageMixDataset and
    plain datasets (e.g. CityscapesSegDataset used by the MTL v2 pipeline).

    Mirrors mmtrack.YOLOXModeSwitchHook but skips update_skip_type_keys when
    the dataset does not support it (i.e. Mosaic was never used to begin with).
    """

    def __init__(self, num_last_epochs: int = 15, skip_type_keys=None):
        self.num_last_epochs = int(num_last_epochs)
        self.skip_type_keys = tuple(
            skip_type_keys or ('Mosaic', 'RandomAffine', 'MixUp'))
        self._restart_dataloader = False

    def before_train_epoch(self, runner) -> None:
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if (epoch + 1) == runner.max_epochs - self.num_last_epochs:
            runner.logger.info('No mosaic and mixup aug now!')
            if hasattr(train_loader.dataset, 'update_skip_type_keys'):
                train_loader.dataset.update_skip_type_keys(self.skip_type_keys)
                if (hasattr(train_loader, 'persistent_workers')
                        and train_loader.persistent_workers):
                    train_loader._DataLoader__initialized = False
                    train_loader._iterator = None
                    self._restart_dataloader = True
            runner.logger.info('Add additional L1 loss now!')
            if hasattr(model, 'detector'):
                model.detector.bbox_head.use_l1 = True
            else:
                model.bbox_head.use_l1 = True
        else:
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True

