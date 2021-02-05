'''
Main function should only be able to see models here. They are models where
we combine recurrent models with prediction head to have more flexibility
over how things are connected.

We can treat the recurrent models as embedding generating model, then the
head models as downstream task model.
'''
import sys
sys.path.append('..')

from constants import *
from .head_models import *
from .recurrent_models import *


class DummyModel(nn.Module):
    def __init__(self, model_args):
        super(DummyModel, self).__init__()
        self.rnn = SimpleLSTM(in_dim=NUM_CHANNELS,
                              hidden_dim=model_args.hidden_size,
                              n_layer=model_args.num_layers,
                              emb_size=model_args.emb_size)
        # self.head = MLP_Regressor(model_args.emb_size)
        self.head = MLP_Classifier(model_args.emb_size, 7)

    def forward(self, x):
        out = self.rnn(x)
        out = self.head(out)
        return out

