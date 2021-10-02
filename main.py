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
from code.statistic import CoOccurStatistic, Ranker


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
    help="preprocess|tain|merge|statistic|indicator"
)
parser.add_argument(
    "--sub_mode",
    "-s",
    type=str,
    default="evaluate",
    help="train|evaluate for indicator mode"
)
args = parser.parse_args()

BackgroundGenerator = lambda x: x


class Runner:
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.hyper = Hyper(os.path.join("config", exp_name + '.json'))
        self.model_dir = os.path.join("saved_models", self.exp_name)
    
    def run(self, mode: str, **kwargs) -> None:
        self._init_logger(mode, **kwargs)

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
        elif mode == 'indicator':
            self.hyper.vocab_init()
            self.hyper.matrix_init()
            self._init_loader()
            self._init_model()
            self._init_optimizer()
            if kwargs["sub_mode"] == 'train':
                self._train_and_evaluate_indicator()
            else:
                self._evaluate_indicator()
        elif mode == 'rank':
            self.hyper.vocab_init()
            self._rank(kwargs["sub_mode"])
        else:
            raise ValueError("Invalid mode!")

    def _init_logger(self, mode, **kwargs):
        log_filename = mode if len(kwargs) == 0 else "-".join([mode] + [str(val) for val in kwargs.values()])
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
        for epoch in range(self.hyper.epoch_num):
            self._train_one_epoch(train_loader, epoch)

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

    def _train_one_epoch(self, train_loader, epoch):
        batch_num = len(train_loader)
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

    def _load_datasets(self):
        logging.info("Load dataset.")
        train_loader = self._get_loader(self.hyper.train, self.hyper.batch_size_train, 8)
        logging.info('Load trainset done.')
        dev_loader = self._get_loader(self.hyper.dev, self.hyper.batch_size_eval, 4)
        logging.info('Load devset done.')
        test_loader = self._get_loader(self.hyper.test, self.hyper.batch_size_eval, 4)
        logging.info('Load testset done.')
        return train_loader,dev_loader,test_loader

    def _get_loader(self, dataset: str, batch_size: int, num_workers: int):
        data_set = self.Dataset(self.hyper, dataset)
        return self.Loader(
            data_set,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers
        )

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
            os.path.join(self.model_dir, self.model.__class__.__name__ + "_" + name),
        )
        self.model.save()

    def load_model(self, name: str):
        """Load model from file.

        Parameters
        ----------
        name : str
            model name.
        """
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_dir, self.model.__class__.__name__ + "_" + name))
        )

    def _statistic(self):
        co_occur_matrix = CoOccurStatistic(self.hyper)
        formatted_matrix = co_occur_matrix.format_co_occur_matrix()
        logging.info(
            "%s co-occur matrix: %s" 
            % (self.hyper.statistic, formatted_matrix)
        )
        co_occur_matrix.save_matrix(os.path.join(self.hyper.data_root, 'co_occur_matrix.json'))

    def _train_and_evaluate_indicator(self):
        logging.info("Load dataset.")
        train_loader = self._get_loader(self.hyper.statistic, self.hyper.batch_size_train, 8)
        test_loader = self._get_loader(self.hyper.statistic, self.hyper.batch_size_eval, 4)
        logging.info("Train start.")
        for epoch in range(self.hyper.epoch_num):
            self._train_one_epoch(train_loader, epoch)
            self._report_indicator(test_loader)

    def _report_indicator(self, loader) -> None:
        self.model.reset()
        self.model.reset_indicators()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        with torch.no_grad():
            for _, sample in pbar:
                output = self.model(sample, is_train=False)
                self.model.update_indicators(sample, output["probability"])
            
        F1_report, role_indicator, NonRole_indicator = self.model.report()

        format_report = lambda result_class, result: "%-6s" % str(result_class) + ", ".join(
                [
                    "%s: %.4f" % (name, value)
                    for name, value in result.items()
                ]
            )

        F1_log = format_report('micro', F1_report[0])
        for i in range(1, len(F1_report)):
            F1_log += '\n' + format_report(i, F1_report[i])
        logging.info(F1_log)

        role_log = 'Each role indicator:'
        for i in range(len(role_indicator)):
            role_log += '\n' + format_report(i, role_indicator[i])
        logging.info(role_log)

        # NonRole_log = 'Only NonRole:\n' + format_report('NonRole', NonRole_indicator[0])
        # logging.info(NonRole_log)
    
    def _evaluate_indicator(self) -> None:
        self.load_model("best")
        logging.info("Load dataset.")
        test_loader = self._get_loader(self.hyper.dev, self.hyper.batch_size_eval, 4)
        logging.info("Evaluate start.")
        self._report_indicator(test_loader)

    def _rank(self, sub_mode: str) -> None:
        dir_name = os.path.join("logs", self.exp_name)
        add_dir = lambda x: os.path.join(dir_name, x)
        log_filename = add_dir('indicator-' + sub_mode + '.log')
        pic_filename = add_dir('Rank.png')

        ranker = Ranker(self.hyper.role_vocab_size)
        ranker.match_file(log_filename)
        ranker.ranking()
        # ranker.save_as_img(pic_filename)
        ranker.save_into_log()
            


if __name__ == '__main__':
    runner = Runner(exp_name=args.exp_name)
    runner.run(mode=args.mode, sub_mode=args.sub_mode)