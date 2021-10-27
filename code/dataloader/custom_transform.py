import numpy as np

from learn2learn.data.transforms import TaskTransform
from learn2learn.data.task_dataset import DataDescription


class BiasedSamplingNWays(TaskTransform):
    def __init__(self, dataset, n, probability):
        super(BiasedSamplingNWays, self).__init__(dataset)
        self.n = n
        self.indices_to_labels = dict(dataset.indices_to_labels)
        self.probability = probability

    def new_task(self):  # Efficient initializer
        labels = self.dataset.labels
        task_description = []
        labels_to_indices = dict(self.dataset.labels_to_indices)
        classes = np.random.choice(len(labels), size=self.n, replace=False, p=self.probability)
        print(classes)
        for cl in classes:
            for idx in labels_to_indices[cl]:
                task_description.append(DataDescription(idx))
        return task_description
    
    def __call__(self, task_description):
        if task_description is None:
            # print("No task_description")
            return self.new_task()
        classes = []
        result = []
        set_classes = set()
        for dd in task_description:
            set_classes.add(self.indices_to_labels[dd.index])
        classes = list(set_classes)
        class_indices = np.random.choice(len(classes), size=self.n, replace=False, p=self.probability)
        classes = [classes[i] for i in class_indices]
        for dd in task_description:
            if self.indices_to_labels[dd.index] in classes:
                result.append(dd)
        return result