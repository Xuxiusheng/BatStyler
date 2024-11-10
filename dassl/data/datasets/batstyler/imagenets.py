import os.path as osp
from dassl.utils import listdir_nohidden
from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase, SFDatum
import glob


@DATASET_REGISTRY.register()
class ImageNetS(DatasetBase):
    dataset_dir = "imagenet-sketch"
    def __init__(self, cfg, train_data):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        test_datasets = []
        self.train_data = train_data
        self.cfg = cfg

        train = self.init_train_data()

        test_datasets.append(self.read_data(self.dataset_dir))

        super().__init__(train_x=train, test=test_datasets)
    
    def read_data(self, dataset_dir):
        # def _load_data_from_directory(directory):
        #     folders = listdir_nohidden(directory)
        #     folders = sorted(folders, key=str.lower)
        #     items_ = []

        #     for label, folder in enumerate(folders):
        #         impaths = glob.glob(osp.join(directory, folder, "*.jpg"))

        #         for impath in impaths:
        #             items_.append((impath, label))
        #     return items_
        items = []

        # impath_label_list = _load_data_from_directory(dataset_dir)
        # for impath, label in impath_label_list:
        #     class_name = impath.split("/")[-2].lower()
        #     item = Datum(
        #         impath=impath,
        #         label=label,
        #         domain="all",
        #         classname=class_name, 
        #     )
        #     items.append(item)

        # return items
        class2id = {}
        with open(osp.join(dataset_dir, 'classnames.txt'), 'r') as fr:
            lines = fr.readlines()
        for line in lines:
            line = line.strip().split(' ')
            class_id = line[0]
            classname = ' '.join(line[1:])
            class2id[classname] = class_id

        classnames = self.train_data["classnames"]
        for label, c in enumerate(classnames):
            class_id = class2id[c]
            impaths = glob.glob(osp.join(dataset_dir, "images", class_id, "*.JPEG"))
            for impath in impaths:
                item = Datum(
                    impath=impath, 
                    label=label, 
                    domain='all', 
                    classname=c
                )
                items.append(item)
        return items


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