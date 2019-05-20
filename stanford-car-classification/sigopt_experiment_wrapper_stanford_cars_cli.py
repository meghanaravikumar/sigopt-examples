from resnet import get_pretrained_resnet
import logging
from enum import Enum
from torch.utils.data import DataLoader
import torch
from resnet import PalmNet
import orchestrate.io
import numpy as np
import math
from resnet_stanford_cars_cli import StanfordCarsCLI, Hyperparameters, CLI


class SigoptExperimentCLI(StanfordCarsCLI):

    def __init__(self):
        super.__init__()

    def load_datasets(self, parsed_cli_arguments):
        return super().load_datasets(parsed_cli_arguments)

    def setup_sigopt_experiment(self):
        """Set up SigOpt Experiment with properties for parameters"""
        pass

    def run_sigopt_experiment(self):
        """Get Sigopt Suggesstion and call run using suggested arguments"""
        pass

    def run(self, sigopt_suggestion_arugments, training_data, validation_data):

        logging.info("loading pretrained model and establishing model characteristics")

        resnet_pretrained_model = get_pretrained_resnet(sigopt_suggestion_arugments[CLI.FREEZE_WEIGHTS.value],
                                                        sigopt_suggestion_arugments[CLI.NUM_CLASSES.value],
                                                        sigopt_suggestion_arugments[CLI.MODEL.value])
        cross_entropy_loss = torch.nn.CrossEntropyLoss()

        sgd_optimizer = torch.optim.SGD(resnet_pretrained_model.parameters(),
                                        lr=sigopt_suggestion_arugments[Hyperparameters.LEARNING_RATE.value],
                                        momentum=sigopt_suggestion_arugments[Hyperparameters.MOMENTUM.value],
                                        weight_decay=sigopt_suggestion_arugments[Hyperparameters.WEIGHT_DECAY.value],
                                        nesterov=sigopt_suggestion_arugments[Hyperparameters.NESTEROV.value])
        learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, mode='min',
                                                                             factor=sigopt_suggestion_arugments[Hyperparameters.LEARNING_RATE_SCHEDULER.value],
                                                                             patience=sigopt_suggestion_arugments[Hyperparameters.SCEDULER_RATE.value],
                                                                             verbose=True)

        logging.info("training model")
        palm_net = PalmNet(epochs=sigopt_suggestion_arugments[CLI.EPOCHS.value], gd_optimizer=sgd_optimizer, model=resnet_pretrained_model,
                           loss_function=cross_entropy_loss,
                           learning_rate_scheduler=learning_rate_scheduler,
                           validation_frequency=sigopt_suggestion_arugments[CLI.VALIDATION_FREQUENCY.value],
                           torch_checkpoint_location=sigopt_suggestion_arugments[CLI.CHECKPOINT.value],
                           model_checkpointing=sigopt_suggestion_arugments[CLI.CHECKPOINT_FREQUENCY.value])

        trained_model, validation_metric = palm_net.train_model(training_data=DataLoader(training_data,
                                                                                         batch_size=sigopt_suggestion_arugments[Hyperparameters.BATCH_SIZE.value],
                                                                                         shuffle=True),
                                                                validation_data=DataLoader(validation_data,
                                                                                           batch_size=sigopt_suggestion_arugments[Hyperparameters.BATCH_SIZE.value],
                                                                                           shuffle=True),
                                                                number_of_labels=sigopt_suggestion_arugments[
                                                                    CLI.NUM_CLASSES.value])
        return trained_model, validation_metric


if __name__ == "__main__":
    sigopt_experiment_cli = SigoptExperimentCLI()
    sigopt_experiment_cli.run_sigopt_experiment()