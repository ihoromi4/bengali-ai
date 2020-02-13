from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from catalyst.dl.core import Callback, CallbackOrder, State
from sklearn.metrics import recall_score
import numpy as np


class HMacroAveragedRecall(Callback):
    def __init__(self,
                 input_keys: list = ["grapheme_root", "consonant_diacritic", "vowel_diacritic"],
                 output_keys: list = ["logit_grapheme_root", "logit_consonant_diacritic", "logit_vowel_diacritic"],
                 weights: list = [2, 1, 1],
                 prefix: str = "hmar"):
        
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.weights = weights
        self.prefix = prefix

        super().__init__(CallbackOrder.Metric)

    def on_batch_end(self, state: State):
        def get_output(key):
            x = state.output[key]
            x = F.softmax(x, 1)
            _, x = torch.max(x, 1)
            return x.detach().cpu().numpy()
        
        inputs = [state.input[key].detach().cpu().numpy() for key in self.input_keys]
        outputs = [get_output(key) for key in self.output_keys]
        scores = [recall_score(input_, output, average='macro') for input_, output in zip(inputs, outputs)]
        final_score = np.average(scores, weights=self.weights)
        
        state.metric_manager.add_batch_value(name=self.prefix, value=final_score)

