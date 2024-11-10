import os.path as osp

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase, SFDatum


@DATASET_REGISTRY.register()
class DomainNetv2(DatasetBase):
    """DomainNet.

    Statistics:
        - 6 distinct domains: Clipart, Infograph, Painting, Quickdraw,
        Real, Sketch.
        - Around 0.6M images.
        - 345 categories.
        - URL: http://ai.bu.edu/M3SDA/.

    Special note: the t-shirt class (327) is missing in painting_train.txt.

    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain
        Adaptation. ICCV 2019.
    """

    dataset_dir = "domainnet"
    domains = [
        "none","clipart", "infograph", "painting", "quickdraw", "real", "sketch"
    ]

    def __init__(self, cfg, train_data):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_dir = osp.join(self.dataset_dir, "splits")
        self.train_data = train_data

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        test_datasets=[]
        for domain in cfg.DATASET.TARGET_DOMAINS:
            test_datasets.append(self._read_data([domain], split="test"))
        train=self.init_train_data()

        super().__init__(train_x=train, test=test_datasets)
        

    def init_train_data(self):
        train_data = self.train_data
        items = []
        classnames = train_data["classnames"]
        n_styles = train_data["n_styles"]
        for label, c in enumerate(classnames):
            for s in range(n_styles):
                item = SFDatum(
                    cls=label, 
                    style=s, 
                    label=label, 
                    classname=c, 
                )
                items.append(item)
        return items

    def _read_data(self, input_domains, split="train"):
        items = []

        for domain, dname in enumerate(input_domains):
            filename = dname + "_" + split + ".txt"
            split_file = osp.join(self.split_dir, filename)

            with open(split_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    impath, label = line.split(" ")
                    classname = impath.split("/")[1]
                    impath = osp.join(self.dataset_dir, impath)
                    label = int(label)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=classname
                    )
                    items.append(item)

        return items
