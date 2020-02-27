from typing import Any, List, Optional, Union
import numpy as np
import torch
from sklearn.metrics import recall_score
import catalyst
from catalyst.dl.core import MetricCallback, LoggerCallback, State

__all__ = [
    'HMacroAveragedRecall',
    'AverageMetric',
]


class HMacroAveragedRecall(MetricCallback):
    def __init__(self, 
                 prefix: str = 'hmar',
                 input_key: str = "targets",
                 output_key: str = "logits",):
        
        super().__init__(prefix, self.metric_fn, input_key, output_key)
        
    def metric_fn(self, outputs, target, **kwargs):
        outputs = torch.max(outputs, dim=-1)[1].detach().cpu().numpy()
        target = target.detach().cpu().unsqueeze(-1).numpy()
        return recall_score(target, outputs, average='macro', zero_division=0)


class AverageMetric(LoggerCallback):
    def __init__(self, prefix: str, metrics: list, weights: list = None):
        super().__init__()
        
        self.prefix = prefix
        self.metrics = metrics
        self.weights = weights
    
    def on_batch_end(self, state: State):
        scores = [state.metric_manager.batch_values[name] for name in self.metrics]
        final_score = np.average(scores, weights=self.weights)
        state.metric_manager.add_batch_value(name=self.prefix, value=final_score)

catalyst.core.registry.Callback(HMacroAveragedRecall)
catalyst.core.registry.Callback(AverageMetric)

