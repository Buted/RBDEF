import os
import logging
# import warnings
import argparse
import torch

import numpy as np
import learn2learn as l2l

from tqdm import tqdm
from typing import Tuple
from functools import partial
from torch.optim import Adam, SGD

from code.config import Hyper
from code.models.coarse_selector import CoarseSelector
from code.preprocess import * 
from code.dataloader import *
from code.models import *
from code.statistic import *


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
parser.add_argument(
    "--load",
    "-l",
    type=bool,
    default=False,
    help="whether or not load modules for training"
)
args = parser.parse_args()

BackgroundGenerator = lambda x: x


class Runner:
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.hyper = Hyper(os.path.join("config", exp_name + '.json'))
        self.model_dir = os.path.join("saved_models", self.exp_name)
        self._set_seed()
    
    def run(self, mode: str, **kwargs) -> None:
        self._init_logger(mode, **kwargs)

        if mode == 'preprocess':
            preprocessor = ACE_Preprocessor(self.hyper)
            preprocessor.gen_type_vocab()
            preprocessor.gen_all_data()
        elif mode == 'divide':
            self.hyper.vocab_init()
            divider = DatasetDivider(self.hyper)
            divider.divide()
        elif mode == 'train':
            self.hyper.vocab_init()
            self._init_loader()
            self._init_model()
            self._init_optimizer()
            if kwargs["load"]:
                self.model.load()
            self.train()
        elif mode == 'meta':
            self.hyper.vocab_init()
            self._init_loader()
            self._init_model()
            self._init_optimizer()
            self._meta_train()
        elif mode == 'evaluate':
            self.hyper.vocab_init()
            self._init_loader()
            self._init_model()
            if kwargs["load"]:
                self.model.load()
            else:
                self.load_model(kwargs["sub_mode"])
            self._evaluate()
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
                self.load_model(kwargs["sub_mode"])
                self._evaluate_indicator()
        elif mode == 'rank':
            self.hyper.vocab_init()
            self._rank(kwargs["sub_mode"])
        elif mode == 'P-R':
            self.hyper.vocab_init()
            self._init_loader()
            self._init_model()
            if kwargs["load"]:
                self.model.load()
            else:
                self.load_model(kwargs["sub_mode"])
            self._plot_pr_curve()
        elif mode == 'save':
            self.hyper.vocab_init()
            self._init_model()
            self.load_model(kwargs["sub_mode"])
            self.model.save()
        elif mode == 'threshold':
            self.hyper.vocab_init()
            self._init_loader()
            self._init_model()
            self._search_threshold()
            self.model.threshold = 0.5
            self.save_model("final")
        elif mode == 'number':
            self.hyper.vocab_init()
            self._role_count()
        elif mode == 'group':
            self.hyper.vocab_init()
            self._group_split()
        elif mode == 'important':
            self.hyper.vocab_init()
            self._compute_important_indicator()
        elif mode == 'roc':
            self.hyper.vocab_init()
            self._init_loader()
            self._init_model()
            self._plot_roc()
        elif mode == 'fewshot':
            self.hyper.vocab_init()
            self._init_loader()
            self._init_model()
            self._init_optimizer()
            self._fewshot_finetune()
        elif mode == 'parameter':
            self.hyper.vocab_init()
            self._init_model()
            logging.info("Model parameter: %f M" % self.model.get_parameter_number())
        else:
            raise ValueError("Invalid mode!")

    def _fewshot_finetune(self):
        # self.model.load()
        self.save_model('fewshot')
        divider = FewshotDivider(self.hyper)
        avg_acc = 0.0
        for i in range(self.hyper.num_task):
            divider.generate_dataset()
            train_loader = self._get_loader('fewshot-train.json', 25, 8)
            test_loader = self._get_loader('fewshot-test.json', self.hyper.batch_size_eval, 4)
            # self.model.load()
            self.load_model('fewshot')
            self.model.train()
            self.model.encoder.eval()
            for epoch in range(self.hyper.epoch_num):
                for trainset in train_loader:
                    self.optimizer.zero_grad()
                    output = self.model(trainset, is_train=True)
                    loss = output["loss"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    self.optimizer.step()
                    # logging.info("Step: %d, loss: %.4f" % (epoch, loss.item()))
            
            self.model.reset()
            for testset in test_loader:
                output = self.model(testset)
            acc = self.model.accuracy.get_metric()['accuracy']
            logging.info("Experiment %d, acc: %.4f" % 
            (i, acc))
            avg_acc += acc
        logging.info("Avgerage acc: %.2f" % (100 * avg_acc / self.hyper.num_task))
            
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
        dataset = {
            "Main model": ACE_Dataset,
            "Simple": ACE_Dataset,
            "Selector": partial(Selector_Dataset, select_roles=self.hyper.meta_roles),
            "Routing": partial(Selector_Dataset, select_roles=self.hyper.meta_roles),
            "Coarse": partial(CoarseSelector_Dataset, select_roles=self.hyper.meta_roles),
            "NonRole": NonRole_Dataset,
            "Branch": partial(Branch_Dataset, select_roles=self.hyper.meta_roles),
            "Meta": Meta_Dataset,
            "FewRoleWithOther": partial(FewRoleWithOther_Dataset, select_roles=self.hyper.meta_roles),
            "FewRole": partial(FewRole_Dataset, select_roles=self.hyper.meta_roles),
            "Head": partial(HeadRole_Dataset, select_roles=self.hyper.meta_roles),
            "HeadWithoutRecall": partial(HeadWithoutRecallRole_Dataset, select_roles=self.hyper.meta_roles),
            "Recall": partial(Recall_Dataset, select_roles=self.hyper.meta_roles),
            "Fuse": ACE_Dataset,
            "AEWithSelector": partial(AE_With_Selector_Dataset, select_roles=self.hyper.meta_roles),
            "MetaWithOther": partial(FewRoleWithOther_Dataset, select_roles=self.hyper.meta_roles),
            "AugmentMeta": ACE_Dataset,
            "Fair": Fair_Dataset,
            "Fewshot": partial(FewRole_Dataset, select_roles=self.hyper.meta_roles)
        }
        loader = {
            "Main model": ACE_loader,
            "Simple": ACE_loader,
            "Selector": Selector_loader,
            "Coarse": ACE_loader,
            "NonRole": ACE_loader,
            "Branch": ACE_loader,
            "Meta": Meta_loader,
            "FewRoleWithOther": ACE_loader,
            "FewRole": ACE_loader,
            "Head": ACE_loader,
            "Recall": ACEWithMeta_loader,
            "Fuse": ACE_loader,
            "AEWithSelector": ACE_With_Selector_loader,
            "MetaWithOther": ACE_loader,
            "AugmentMeta": ACE_loader,
            "Fair": Fair_loader
        }
        self.Dataset = dataset.get(self.hyper.model, ACE_Dataset)
        self.Loader = loader.get(self.hyper.model, ACE_loader)

    def _init_model(self):
        logging.info(self.hyper.model)
        model_dict = {
            "Main model": AEModel,
            "Simple": SimpleAEModel,
            "Selector": Selector,
            "Routing": Routing,
            "Coarse": CoarseSelector,
            "NonRole": NonRoleFilter,
            "Branch": BranchSelector,
            "Meta": MetaAEModel,
            "FewRoleWithOther": MetaAEModel,
            "FewRole": MetaAEModel,
            "Head": HeadAEModel,
            "HeadWithoutRecall": HeadWithoutRecallAEModel,
            "Recall": RecallAEModel,
            "Fuse": FusedAEModel,
            "FuseWithMultiEncoders": FusedAEWithMultiEncoders,
            "AEWithSelector": AEWithSelector,
            "MetaWithOther": MetaWithOtherAEModel,
            "AugmentMeta": AugmentMetaAEModel,
            "Dice": DiceAEModel,
            "ClassBalanced": ClassBalancedAEModel,
            "TDE": TDEAEModel,
            "Fair": FairAEModel,
            "Fewshot": MetaAEModel
        }
        self.model = model_dict[self.hyper.model](self.hyper)

    def _init_optimizer(self):
        if self.hyper.model in ["Main model", "AEWithSelector", "MetaWithOther", "Dice", "ClassBalanced", "TDE", "Fair"]:
            bert_params = list(map(id, self.model.encoder.encoder.parameters()))
            scratch_params = filter(lambda p: id (p) not in bert_params, self.model.parameters())
            params_with_lr = [
                {'params': self.model.encoder.encoder.parameters(), 'lr': 1e-5},
                {'params': scratch_params, 'lr': self.hyper.lr}
            ]
        else:
            params_with_lr = [
                {'params': self.model.classifier.parameters(), 'lr': self.hyper.lr}
            ]
        m = {"adam": Adam(params_with_lr), "sgd": SGD(params_with_lr, lr=self.hyper.lr)}
        self.optimizer = m[self.hyper.optimizer]
        # torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, gamma=0.7, last_epoch=-1)

    def train(self):
        train_loader, dev_loader, test_loader = self._load_datasets()
        score = 0.
        best_epoch = 0
        logging.info("Train start.")
        for epoch in range(self.hyper.epoch_num):
            self._train_one_epoch(train_loader, epoch)

            new_score, log = self.evaluation(dev_loader)
            logging.info(log)
            # if self.hyper.model == "Selector":
                # self.save_model(str(epoch))
            # elif self.hyper.model == "FewRole":
                # self.save_model(str(epoch))
            if new_score >= score:
                score = new_score
                best_epoch = epoch
                self.save_model("best")
            self._evaluate(test_loader)

        logging.info("best epoch: %d \t F1 = %.2f" % (best_epoch, score))
        logging.info("Evaluate on testset:")
        self.load_model("best")
        self._evaluate(test_loader)

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

    def _meta_train(self):
        meta_model = l2l.algorithms.MAML(self.model.classifier, self.hyper.meta_lr, allow_nograd=True)
        torch.optim.lr_scheduler.StepLR(self.optimizer, 200, gamma=0.7, last_epoch=-1)

        # load data
        self.model.train()
        self.model.encoder.eval()
        train_dataloader_generator = self._get_meta_dataset(
            self.hyper.train, 
            n=self.hyper.n, k=self.hyper.k, 
            num_tasks=self.hyper.meta_steps*self.hyper.num_task
        )
        sampler = lambda dataset: iter(self.Loader(
                dataset,
                batch_size=self.hyper.n*self.hyper.k,
                pin_memory=False,
                shuffle=True
            )).next()
        # dev_dataloader_generator = self._get_meta_dataset(self.hyper.dev, self.hyper.dev_filter_relations, n=12, k=2)
        logging.info("Meta training start.")
        for epoch in range(self.hyper.meta_steps):
            self.optimizer.zero_grad()
            loss = 0
            for _ in range(self.hyper.num_task):
                cloned_model = meta_model.clone()
                support_dataset, query_dataset = next(train_dataloader_generator)
                for _ in range(self.hyper.fast_steps):
                    sample = sampler(support_dataset)
                    output = self.model.meta_forward(sample, cloned_model, support_dataset.remap)
                    cloned_model.adapt(output["loss"])
                    # logging.info("loss: %.4f" % output["loss"].item())
                
                sample = sampler(query_dataset)
                output = self.model.meta_forward(sample, cloned_model, query_dataset.remap, outloop=True)
                loss += output["loss"]
                
            if np.isnan(loss.item()):
                logging.info("At step %d:Loss becomes nan, stopping traning." % epoch)
                break

            loss /= self.hyper.num_task
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            logging.info("Step: %d, loss: %.4f" % (epoch, loss.item()))

            if (epoch + 1) % 300 == 0:
                self.save_model(str(epoch + 1))

        self.save_model("meta")
        logging.info("Meta training done.")

    def _get_meta_dataset(self, dataset_name: str, n: int=5, k: int=2, num_tasks: int=1100):
        logging.info("Load Meta Trainset.")
        filter_roles = list(range(self.hyper.role_vocab_size))
        for role in self.hyper.filter_roles:
            filter_roles.remove(role)
        # biased_dataset = AugmentedBiasedSampling_Dataset(self.hyper, dataset_name)
        biased_dataset = ImportantIndicator_Dataset(self.hyper, dataset_name)
        # biased_dataset = Meta_Dataset(self.hyper, dataset_name)
        dataset = l2l.data.MetaDataset(biased_dataset, indices_to_labels=biased_dataset.indices2labels)
        # print({label: len(indices) for label, indices in dataset.labels_to_indices.items()})
        
        yield from self._yield_metaset(n, k, num_tasks, dataset, filter_roles)

    def _yield_metaset(self, n, k, num_tasks, dataset, filter_roles):
        tasks = l2l.data.TaskDataset(dataset,
            task_transforms=[
                # l2l.data.transforms.FilterLabels(dataset, filter_roles),
                # BiasedSamplingNWays(dataset, n=n, probability=biased_dataset.probability),
                # l2l.data.transforms.KShots(dataset, k=k*2),
                l2l.data.transforms.FusedNWaysKShots(dataset, n=n, k=k*2, filter_labels=filter_roles)
            ],
            num_tasks=num_tasks
        )
        for _ in range(num_tasks):
            task = tasks.sample()
            support_indices = np.zeros(task.shape[0], dtype=bool)
            support_indices[np.arange(n*k) * 2] = True
            query_indices = torch.from_numpy(~support_indices)
            support_indices = torch.from_numpy(support_indices)
            support_indices = task[support_indices]
            query_indices = task[query_indices]
            support_dataset = FewShot_Dataset(dataset, support_indices)
            query_dataset = FewShot_Dataset(dataset, query_indices, support_dataset.remap)
            yield (support_dataset, query_dataset)

    def _load_datasets(self):
        logging.info("Load dataset.")
        train_loader = self._get_train_loader(self.hyper.train, self.hyper.batch_size_train, 8)
        logging.info('Load trainset done.')
        dev_loader = self._get_loader(self.hyper.dev, self.hyper.batch_size_eval, 4)
        logging.info('Load devset done.')
        test_loader = self._get_loader(self.hyper.test, self.hyper.batch_size_eval, 4)
        logging.info('Load testset done.')
        return train_loader,dev_loader,test_loader

    def _get_train_loader(self, dataset: str, batch_size: int, num_workers: int):
        if self.hyper.model == "FewRoleWithOther":
            train_set = MetaFewRoleWithOther_Dataset(self.hyper, dataset, select_roles=self.hyper.meta_roles)
        elif self.hyper.model == "FewRole":
            train_set = MetaFewRole_Dataset(self.hyper, dataset, select_roles=self.hyper.meta_roles)
        elif self.hyper.model == "MetaWithOther":
            train_set = MetaFewRoleWithOther_Dataset(self.hyper, dataset, select_roles=self.hyper.meta_roles)
            return Balanced_loader(train_set, self.hyper)
        elif self.hyper.model == "AugmentMetaAEModel":
            train_set = Meta_Dataset(self.hyper, dataset)
            return BalancedWithOther_loader(train_set, self.hyper)
        else:
            train_set = self.Dataset(self.hyper, dataset)
        sampler = WeightedRoleSampler(train_set).sampler if self.hyper.model in ["Selector", "FewRoleWithOther"] else None
        return self.Loader(
            train_set,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers
        ) if self.hyper.model not in ["FewRoleWithOther", "FewRole"] else Balanced_loader(train_set, self.hyper)

    def _get_loader(self, dataset: str, batch_size: int, num_workers: int):
        data_set = self.Dataset(self.hyper, dataset)
        return self.Loader(
            data_set,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers
        )

    def _search_threshold(self):
        dev_loader = self._get_loader(self.hyper.dev, self.hyper.batch_size_eval, 4)
        test_loader = self._get_loader(self.hyper.test, self.hyper.batch_size_eval, 4)
        logging.info('Load testset done.')
        threshold = 0.5
        plus = [0]
        best_f1 = 0.0
        best_threshold = 0.
        for plus_thresold in plus:
            threshold += plus_thresold
            self.model.threshold = threshold
            logging.info("Threshold: %.3f" % threshold)
            f1 = self._evaluate(dev_loader)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
            self._evaluate(test_loader)
        logging.info("Best threshold: %.3f" % best_threshold)
        self.model.threshold = best_threshold
        self._evaluate(test_loader)

    def _evaluate(self, test_loader=None):
        if test_loader is None:
            test_loader = self._get_loader(self.hyper.test, self.hyper.batch_size_eval, 4)
            logging.info('Load testset done.')

        fscore, _ = self.evaluation(test_loader)

        F1_report = self.model.metric.report_all()
        format_report = self.report_format()

        F1_log = format_report('avg', F1_report[0])
        for i in range(1, len(F1_report)):
            F1_log += '\n' + format_report(i, F1_report[i])
        logging.info(F1_log)
        return fscore

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
        co_occur_matrix.save()
        co_occur_matrix.reverse_save()

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

        format_report = self.report_format()

        F1_log = format_report(self.hyper.metric, F1_report[0])
        for i in range(1, len(F1_report)):
            F1_log += '\n' + format_report(i, F1_report[i])
        logging.info(F1_log)

        role_log = 'Each role indicator:'
        for i in range(len(role_indicator)):
            role_log += '\n' + format_report(i, role_indicator[i])
        logging.info(role_log)

    @staticmethod
    def report_format():
        format_report = lambda result_class, result: "%-6s" % str(result_class) + ", ".join(
                [
                    "%s: %.4f" % (name, value)
                    for name, value in result.items()
                ]
            )
            
        return format_report
    
    def _evaluate_indicator(self) -> None:
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
        ranker.save_into_log()
    
    def _plot_pr_curve(self):
        dev_loader = self._get_loader(self.hyper.dev, self.hyper.batch_size_eval, 4)
        self.evaluation(dev_loader)
        self.model.curve.get_metric(True)

    def _role_count(self):
        counter = RoleNumberCounter(self.hyper.train, self.hyper)
        counter.count()
    
    def _group_split(self):
        spliter = GroupSplit(self.hyper.train, self.hyper)
        spliter.split()

    def _compute_important_indicator(self):
        co_occur_matrix = CoOccurStatistic(self.hyper)
        co_occur_matrix.save_important_indicator_of_samples()

    def _plot_roc(self):
        test_loader = self._get_loader(self.hyper.test, self.hyper.batch_size_eval, 4)
        logging.info('Load testset done.')
        self.evaluation(test_loader)
        self.model.plot()

    def _set_seed(self):
        np.random.seed(self.hyper.seed)
        torch.manual_seed(self.hyper.seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False
            


if __name__ == '__main__':
    runner = Runner(exp_name=args.exp_name)
    runner.run(mode=args.mode, sub_mode=args.sub_mode, load=args.load)