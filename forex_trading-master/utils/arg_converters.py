'''
Created for Winter 2019 Stanford CS224W
Jingbo Yang, Ruge Zhao, Meixian Zhu
Pytorch-specific implementation

Adapted from Driver2vec as part of CS341
Adapted from Stanford AI for Healthcare Bootcamp deep learning infrastructure
'''


import argparse


def str_to_bool(arg):
    """Convert an argument string into its boolean value.

    Args:
        arg (string): String representing a bool.

    Returns:
        (bool) Boolean value for the string.
    """
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
