import time
import os
import torch
from ppcls.models import build_model
from ppcls.core import build_optimizer, build_loader
from ppcls.core.utils import show_config_values, AverageMeter

class SimpleTrainer:
    """
    一个简单的训练器，用于卷积神经网络的训练，工作流程如下：
    before_train -> before_epoch -> before_iter -> run_iter -> after_iter -> after_epoch -> after train
    """
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.model = None
        self.optimizer = None
        self.loss = None
        self.train_loader = None
        self.val_loader = None

        # 设置训练的设备
        if torch.cuda.is_available():
            self.device = self.cfg.get("gpu", "cuda:0")
        else:
            self.device = "cpu"

        self.curr_epoch = 1 # 训练时，当前的epoch
        self.curr_iter = 0  # 训练时，当前的第几次迭代
        self.best_acc = 0   # 记录最好的准确率

    def train(self):
        """
        训练卷积神经网络主要流程
        """
        self.before_train()
        for epoch in range(self.cfg["max_epoch"]):
            self.before_train_epoch()
            for image, label in self.train_loader:
                # 迭代一次数据之前的操作
                self.before_train_iter()
                # 对一批数据进行前向传播和反向传播
                self.run_iter(image, label)
                # 对一批数据进行迭代后处理
                self.after_train_iter()
            self.after_train_epoch()
        self.after_train()

    def before_train(self):
        """
        在训练卷积神经网络之前要做的工作
        """
        # 显示配置文件信息
        self.logger.info("-"*30+"config info"+"-"*30)
        show_config_values(self.cfg,logger=self.logger.info)
        self.logger.info("-"*30+"run before train"+"-"*30)
        # 创建数据集加载器
        self.train_loader, self.val_loader = build_loader(self.cfg["loader"])
        self.logger.info("build data loader success")
        # 创建网络模型
        self.model = build_model(self.cfg["model"]).to(self.device)
        self.logger.info("build model success")
        # 创建优化器
        self.optimizer = build_optimizer(self.cfg["optimizer"],  self.model.parameters())
        self.logger.info("build optimizer success")
        # 创建损失函数
        self.loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.logger.info("build loss success")

    def before_train_epoch(self):
        self.curr_iter = 0
        self.logger.info("-"*30 + f"epoch: {self.curr_epoch}/{self.cfg['max_epoch']}"+"-"*30)
        self.loss_avg = AverageMeter()

    def before_train_iter(self):
        # 优化器 梯度清零
        self.optimizer.zero_grad()
        # 模型进入训练模式
        self.model.train()
        # 记录run_iter开始的时间
        self.iter_start_time = time.time()

    def run_iter(self, images, labels):
        # 判断输入图片和标签是否在同一设备上
        if images.device != self.device: images = images.to(self.device)
        if labels.device != self.device: labels = labels.to(self.device)
        # 前向传播
        predict = self.model(images)        # 前向传播得到结果
        loss = self.loss(predict, labels)   # 根据标签对得到的结果进行损失的计算
        # 反向传播
        loss.backward()
        self.loss_value = float(loss)

    def after_train_iter(self):
        run_iter_time = time.time() - self.iter_start_time
        self.loss_avg.update(self.loss_value)
        if self.curr_iter % 10 == 0:
            msg = f"=> Step {self.curr_iter}/{len(self.train_loader)} \ttime:{int(run_iter_time*1000)}ms\t" \
                  f" loss: {self.loss_avg.avg:0.5f}"
            self.logger.info(msg)

        self.optimizer.step()       # 对网络的参数进行更新
        self.curr_iter += 1
        pass

    def after_train_epoch(self):
        self.curr_epoch +=1
        acc = validate(self.val_loader, self.model, self.device)
        self.logger.info(f"accuracy:\t{acc:0.2f}")

        if self.best_acc < acc:
            save_path = os.path.join(self.cfg["output_dir"], "best.pth")
            torch.save(self.model.state_dict(), save_path)
            self.best_acc = acc

    def after_train(self):
        self.logger.info(f"train finish, the best accuracy is {self.best_acc}")

def validate(val_loader, model, device="cpu"):
    acc_avg = AverageMeter()
    # 验证的时候，不进行求导
    model.eval() # 模型是验证模式
    with torch.no_grad():
        for images, labels in val_loader:
            # 同一计算设备
            if images.device != device: images = images.to(device)
            if labels.device != device: labels = labels.to(device)

            output = model(images)   # 得到结果 output.shape 是 [batch, number_class]
            # 找到number_class中最大的一个索引
            _, index = output.max(dim=-1)
            # 计算准确率
            acc = sum(index == labels) / len(labels)
            acc_avg.update(acc)

    return acc_avg.avg
