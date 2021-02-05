'''
Created for Winter 2019 Stanford CS224W
Jingbo Yang, Ruge Zhao, Meixian Zhu
Pytorch-specific implementation

Adapted from Driver2vec as part of CS341
Adapted from Stanford AI for Healthcare Bootcamp deep learning infrastructure
'''


import datetime
import getpass
import os
from pathlib import Path
import pickle
import sys
import socket

import torch
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from constants import TEMP_FOLDER, GC_HOME
from utils import GCOpen, GCStorage


def find_tf_event(name_suffix):
    for f in os.listdir(TEMP_FOLDER):
        last = f.split('.')[-1]
        if name_suffix == last:
            return f


class Logger(object):
    """Class for logging output."""

    unique_logger = None
    @staticmethod
    def get_unique_logger():
        if Logger.unique_logger is None:
            raise ValueError('Unable to find a unique logger.')
        return Logger.unique_logger

    def __init__(self, level, general_log_path, outputs_folder):
        """Both general_log_ath and ouptuts_folder
           should be in heavy logging.
        """
        self.log_level = level

        # self.general_log_file = general_log_path.open('w')
        self.general_log_file = GCOpen(general_log_path, 'w')
        self.general_log_file.open()

        self.file_outputs_dir = outputs_folder / 'output_files'
        # self.file_outputs_dir.mkdir(exist_ok=True)

        exp_name = str(outputs_folder).split('/')[-1]

        self.summary_writer = SummaryWriter(log_dir=str(TEMP_FOLDER),
                                            filename_suffix='.' + exp_name)
        tf_filename = find_tf_event(exp_name)
        self.sw_local_path = Path(TEMP_FOLDER) / tf_filename
        self.sw_gc_path = outputs_folder / tf_filename

        self.log("Starting new experiment at " +
                 datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.log("User: " + getpass.getuser())
        self.log("Host: " + socket.gethostname())

        Logger.unique_logger = self

    def sync_summary_writer(self):
        try:
            self.summary_writer.flush()
        except:
            pass
        GCStorage.MONO.upload(self.sw_local_path, self.sw_gc_path)

    def log(self, *args):
        """Writes args to the log file."""
        self.log_stdout(*args)
        print(*args, file=self.general_log_file.file)
        self.general_log_file.flush()

    def log_stdout(self, *args, **kwargs):
        """Writes args to the console."""

        do_print = False
        if 'level' in kwargs:
            if kwargs['level'] >= self.log_level:
                do_print = True
            else:
                do_print = False
        else:
            do_print = True

        if do_print:
            print(*args, file=sys.stdout)
            sys.stdout.flush()

    def log_scalars(self, scalar_dict,
                    iterations, steps_per_epoch=None,
                    step_in_epoch=None, cur_epoch=None,
                    print_to_stdout=True):
        """Log all values in a dict as scalars to TensorBoard."""
        if len(scalar_dict) != 0:
            thing1 = f'/{step_in_epoch}/{steps_per_epoch}' \
                                        if step_in_epoch is not None and \
                                           steps_per_epoch is not None \
                                        else ""
            thing2 = f'@Epoch {cur_epoch}' if cur_epoch is not None else ""
            self.log_stdout(f'Step {iterations}{thing1}{thing2} Scalars')

        for k, v in scalar_dict.items():
            if print_to_stdout:
                temp_k = k.replace(':', '/')
                self.log_stdout(f'\t[{temp_k}: {v:.5g}]')
            k = k.replace(':', '/')  # Group in TensorBoard.
            self.summary_writer.add_scalar(k, v, iterations)
            self.sync_summary_writer()

    def log_images(self, image_dict,
                   iterations, step_in_epoch=None, cur_epoch=None,
                   save_to_outputs=True, include_iter=False):
        """Log all images in a dict as images to TensorBoard."""
        if len(image_dict) != 0:
            thing1 = f'/{step_in_epoch}' if step_in_epoch is not None else ""
            thing2 = f'@Epoch {cur_epoch}' if cur_epoch is not None else ""
            self.log_stdout(f'Step {iterations}{thing1}{thing2} Images')

        for k, v in image_dict.items():
            np_image, plt_figure = v
            if save_to_outputs:
                if include_iter:
                    img_name = k.replace(':', '_') + f'_{iterations}.png'
                else:
                    img_name = k.replace(':', '_') + '.png'
                self.log_image(plt_figure, img_name)
            k = k.replace(':', '/')  # Group in TensorBoard.

            self.summary_writer.add_image(k,
                                          np_image,
                                          iterations,
                                          dataformats='HWC')
            self.sync_summary_writer()

    def log_text(self, text_dict,
                iterations, step_in_epoch=None, cur_epoch=None,
                print_to_stdout=True):
        """Log all text in a dict to TensorBoard."""
        if len(text_dict) != 0:
            thing1 = f'/{step_in_epoch}' if step_in_epoch is not None else ""
            thing2 = f'@Epoch {cur_epoch}' if cur_epoch is not None else ""
            self.log_stdout(f'Step {iterations}{thing1}{thing2} Texts')

        for k, v in text_dict.items():
            if print_to_stdout:
                temp_k = k.replace(':', '/')
                self.log_stdout(f'\t[{temp_k}: {v}]')
            k = k.replace(':', '/')  # Group in TensorBoard.

            self.summary_writer.add_text(k,
                                         v,
                                         iterations)
            self.sync_summary_writer()

    def log_numpy(self, output, filename):
        with GCOpen(self.file_outputs_dir / (filename + '.npy'), 'wb') as f:
            np.save(f, output)

    def log_data(self, data, filename):
        # filehandle = (self.file_outputs_dir / (filename + '.pickle'))\
        #                                                       .open('wb')
        with GCOpen(self.file_outputs_dir / (filename + '.pickle'), 'wb') as f:
            pickle.dump(data, f)

    def log_image(self, fig, filename):
        # filehandle = GCOpen(self.file_outputs_dir / filename, 'wb')
        with GCOpen(self.file_outputs_dir / filename, 'wb') as f:
            fig.savefig(f)

    def close(self):
        '''Closes the log file'''
        self.general_log_file.close()
