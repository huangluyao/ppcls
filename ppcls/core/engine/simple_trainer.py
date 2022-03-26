import torch
from ppcls.models import build_model
from ppcls.core import build_optimizer, ClassificationDataset, build_loader


class SimpleTrainer:

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.model = None
        self.optimizer = None
        self.loss = None
        self.train_loader = None
        self.val_loader = None

    def train(self):
        self.before_train()
        for epoch in range(self.cfg["max_epoch"]):
            self.before_train_epoch()
            for image, label in self.train_loader:
                self.before_train_iter()
                self.run_iter()
                self.after_train_iter()
            self.after_train_epoch()
        self.after_train()

    def before_train(self):
        self.logger.info("-"*30+"run before train"+"-"*30)
        self.train_loader, self.val_loader = build_loader(self.cfg["loader"])
        self.logger.info("build data loader success")
        self.model = build_model(self.cfg["model"])
        self.logger.info("build model success")
        self.optimizer = build_optimizer(self.cfg["optimizer"],  self.model.parameters())
        self.logger.info("build optimizer success")
        self.loss = torch.nn.CrossEntropyLoss()
        self.logger.info("build loss success")

    def before_train_epoch(self):
        pass

    def before_train_iter(self):
        pass

    def run_iter(self):
        pass

    def after_train_iter(self):
        pass

    def after_train_epoch(self):
        pass

    def after_train(self):
        pass
