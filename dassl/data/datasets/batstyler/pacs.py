import os.path as osp

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase, SFDatum


@DATASET_REGISTRY.register()
class DensePACS(DatasetBase):
    dataset_dir = "pacs"
    domains = ["none", "art_painting", "cartoon", "photo", "sketch"]
    _error_paths = ["sketch/dog/n02103406_4068-1.png"]

    def __init__(self, cfg, train_data):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")
        self.split_dir = osp.join(self.dataset_dir, "splits")

        self.cfg = cfg
        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        self.train_data = train_data
        train = self.init_train_data()

        test_dataset = []
        for domain in cfg.DATASET.TARGET_DOMAINS:
            test_dataset.append(self._read_data([domain], "test"))

        super().__init__(train_x=train, test=test_dataset)

    def init_train_data(self):
        train_data = self.train_data
        items = []
        classnames = train_data["classnames"]
        n_cls = train_data["n_cls"]
        n_styles = train_data["n_styles"]
        for c in range(n_cls):
            for s in range(n_styles):
                item = SFDatum(
                    cls=c, 
                    style=s, 
                    label=c, 
                    classname=classnames[c], 
                )
                items.append(item)
        return items
    
    def _read_data(self, input_domains, split):
        items = []

        for domain, dname in enumerate(input_domains):
            if split == "all":
                file_train = osp.join(
                    self.split_dir, dname + "_train_kfold.txt"
                )
                impath_label_list = self._read_split_pacs(file_train)
                file_val = osp.join(
                    self.split_dir, dname + "_crossval_kfold.txt"
                )
                impath_label_list += self._read_split_pacs(file_val)
            else:
                file = osp.join(
                    self.split_dir, dname + "_" + split + "_kfold.txt"
                )
                impath_label_list = self._read_split_pacs(file)

            for impath, label in impath_label_list:
                classname = impath.split("/")[-2]
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=dname,
                    classname=classname
                )
                items.append(item)

        return items

    def _read_split_pacs(self, split_file):
        items = []

        with open(split_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                if impath in self._error_paths:
                    continue
                impath = osp.join(self.image_dir, impath)
                label = int(label) - 1
                items.append((impath, label))

        return items