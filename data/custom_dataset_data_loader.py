import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset1(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset1 import AlignedDataset1
        dataset = AlignedDataset1()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'unaligned_random_crop':
        from data.unaligned_random_crop import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'pair':
        from data.pair_dataset import PairDataset
        dataset = PairDataset()
    elif opt.dataset_mode == 'syn':
        from data.syn_dataset import PairDataset
        dataset = PairDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

def CreateDataset2(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset2 import AlignedDataset2
        dataset = AlignedDataset2()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'unaligned_random_crop':
        from data.unaligned_random_crop import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'pair':
        from data.pair_dataset import PairDataset
        dataset = PairDataset()
    elif opt.dataset_mode == 'syn':
        from data.syn_dataset import PairDataset
        dataset = PairDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset1 = CreateDataset1(opt)
        self.dataset2 = CreateDataset2(opt)
        self.dataloader1 = torch.utils.data.DataLoader(
            self.dataset1,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
        self.dataloader2 = torch.utils.data.DataLoader(
            self.dataset2,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader1, self.dataloader2

    def __len__(self):
        return (min(len(self.dataset1), self.opt.max_dataset_size), min(len(self.dataset2), self.opt.max_dataset_size))
