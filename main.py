import os
import logging
# import warnings
import argparse
import torch

from tqdm import tqdm
from typing import Tuple
from torch.optim import Adam, SGD

from code.config import Hyper
from code.preprocess import ACE_Preprocessor, merge_dataset
from code.dataloader import ACE_Dataset, ACE_loader
from code.models import AEModel
from code.statistic import CoOccurStatistic


os.environ["TOKENIZERS_PARALLELISM"] = "True"
# warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_name",
    "-e",
    type=str,
    default="ace",
    help="config/ace.json"
)
parser.add_argument(
    "--mode",
    "-m",
    type=str,
    default="preprocess",
    help="preprocess|tain|evaluation"
)
args = parser.parse_args()

BackgroundGenerator = lambda x: x


class Runner:
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.hyper = Hyper(os.path.join("config", exp_name + '.json'))
        self.model_dir = os.path.join("saved_models", self.exp_name)
    
    def run(self, mode: str) -> None:
        self._init_logger(mode)

        if mode == 'preprocess':
            preprocessor = ACE_Preprocessor(self.hyper)
            preprocessor.gen_type_vocab()
            preprocessor.gen_all_data()
        elif mode == 'train':
            self.hyper.vocab_init()
            self._init_loader()
            self._init_model()
            self._init_optimizer()
            self.train()
        elif mode == 'merge':
            merge_dataset(self.hyper)
        elif mode == 'statistic':
            self.hyper.vocab_init()
            self._statistic()
        else:
            raise ValueError("Invalid mode!")



    def _init_logger(self, mode):
        log_filename = mode
        log_dir = os.path.join("logs", self.exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            filename=os.path.join(log_dir, log_filename + ".log"),
            filemode="w",
            format="%(asctime)s - %(message)s",
            level=logging.INFO
        )

    def _init_loader(self):
        self.Dataset = ACE_Dataset
        self.Loader = ACE_loader

    def _init_model(self):
        logging.info(self.hyper.model)
        self.model = AEModel(self.hyper)

    def _init_optimizer(self):
        bert_params = list(map(id, self.model.encoder.parameters()))
        scratch_params = filter(lambda p: id (p) not in bert_params, self.model.parameters())
        params_with_lr = [
            {'params': self.model.encoder.parameters(), 'lr': 1e-5},
            {'params': scratch_params, 'lr': self.hyper.lr}
        ]
        m = {"adam": Adam(params_with_lr), "sgd": SGD(params_with_lr, lr=0.5)}
        self.optimizer = m[self.hyper.optimizer]

    def train(self):
        train_loader, dev_loader, test_loader = self._load_datasets()
        score = 0.
        best_epoch = 0
        logging.info("Train start.")
        batch_num = len(train_loader)
        for epoch in range(self.hyper.epoch_num):
            self.model.train()
            pbar = tqdm(
                enumerate(BackgroundGenerator(train_loader)),
                total=len(train_loader)
            )
            loss = 0
            for _, sample in pbar:
                self.optimizer.zero_grad()

                output = self.model(sample, is_train=True)

                output["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

                self.optimizer.step()

                pbar.set_description(output["description"](epoch, self.hyper.epoch_num))
                loss += output["loss"].item()
            
            logging.info("Epoch: %d, loss: %.4f" % (epoch, loss / batch_num))
            new_score, log = self.evaluation(dev_loader)
            logging.info(log)
            if new_score >= score:
                score = new_score
                best_epoch = epoch
                self.save_model("best")

        logging.info("best epoch: %d \t F1 = %.2f" % (best_epoch, score))
        logging.info("Evaluate on testset:")
        self.load_model("best")
        _, log = self.evaluation(test_loader)
        logging.info(log)

    def _load_datasets(self):
        logging.info("Load dataset.")
        train_set = self.Dataset(self.hyper, self.hyper.train)
        train_loader = self.Loader(
            train_set,
            batch_size=self.hyper.batch_size_train,
            pin_memory=True,
            num_workers=8
        )
        logging.info('Load trainset done.')
        dev_set = self.Dataset(self.hyper, self.hyper.dev)
        dev_loader = self.Loader(
            dev_set,
            batch_size=self.hyper.batch_size_eval,
            pin_memory=True,
            num_workers=4,
        )
        logging.info('Load devset done.')
        test_set = self.Dataset(self.hyper, self.hyper.test)
        test_loader = self.Loader(
            test_set,
            batch_size=self.hyper.batch_size_eval,
            pin_memory=True,
            num_workers=4,
        )
        logging.info('Load testset done.')
        return train_loader,dev_loader,test_loader

    def evaluation(self, loader) -> Tuple[float, str]:
        self.model.reset()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        with torch.no_grad():
            for _, sample in pbar:
                output = self.model(sample, is_train=False)

        result = self.model.get_metric()

        log = (
            ", ".join(
                [
                    "%s: %.4f" % (name, value)
                    for name, value in result.items()
                    if not name.startswith("_")
                ]
            )
            + " ||"
        )

        return result["fscore"], log

    def save_model(self, name: str):
        """Save model in disk.

        Parameters
        ----------
        name : str
            model name.
        """
        # def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, self.exp_name + "_" + name),
        )

    def load_model(self, name: str):
        """Load model from file.

        Parameters
        ----------
        name : str
            model name.
        """
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_dir, self.exp_name + "_" + name))
        )

    def _statistic(self):
        co_occur_matrix = CoOccurStatistic(self.hyper)
        formatted_matrix = co_occur_matrix.format_co_occur_matrix()
        logging.info(
            "%s co-occur matrix: %s" 
            % (self.hyper.statistic, formatted_matrix)
        )



if __name__ == '__main__':
    runner = Runner(exp_name=args.exp_name)
    runner.run(mode=args.mode)