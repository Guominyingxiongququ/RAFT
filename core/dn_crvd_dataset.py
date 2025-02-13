# Copyright (c) OpenMMLab. All rights reserved.
from base_dn_dataset import BaseDNDataset
from registry import DATASETS


@DATASETS.register_module()
class DNCRVDDataset(BaseDNDataset):
    """CRVD dataset for video denoise.

    The dataset loads several LQ (Low-Quality) frames and a center GT
    (Ground-Truth) frame. Then it applies specified transforms and finally
    returns a dict containing paired data and other information.

    It reads CRVD keys from the txt file.
    Each line contains:
    1. image name; 2, image shape, seperated by a white space.
    Examples:

    ::

        000/00000000.png (720, 1280, 3)
        000/00000001.png (720, 1280, 3)

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        ann_file (str | :obj:`Path`): Path to the annotation file.
        num_input_frames (int): Window size for input frames.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        val_partition (str): Validation partition mode. Choices ['official' or
        'REDS4']. Default: 'official'.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self,
                 lq_folder,
                 predn_folder,
                 gt_folder,
                 ann_file,
                 num_input_frames,
                 pipeline,
                 scale,
                 val_partition='official',
                 test_mode=False):
        super().__init__(pipeline, scale, test_mode)
        assert num_input_frames % 2 == 1, (
            f'num_input_frames should be odd numbers, '
            f'but received {num_input_frames }.')
        self.lq_folder = str(lq_folder)
        self.predn_folder = str(predn_folder)
        self.gt_folder = str(gt_folder)
        self.ann_file = str(ann_file)
        self.num_input_frames = num_input_frames
        self.val_partition = val_partition
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annoations for CRVD dataset.

        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        # get keys
        with open(self.ann_file, 'r') as fin:
            keys = [v.strip().split('.')[0] for v in fin]
        
        if self.val_partition == 'CRVD':
            val_partition = ['007', '008', '009', '010', '011']
        else:
            raise ValueError(
                f'Wrong validation partition {self.val_partition}.'
                f'Supported ones are ["CRVD"]')

        if self.test_mode:
            keys = [v for v in keys if v.split('/')[0] in val_partition]
        else:
            keys = [v for v in keys if v.split('/')[0] not in val_partition]
        
        data_infos = []
        for key in keys:
            data_infos.append(
                dict(
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    key=key,
                    max_frame_num=7, 
                    num_input_frames=self.num_input_frames))

        return data_infos
