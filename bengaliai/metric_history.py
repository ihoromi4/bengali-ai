from catalyst.dl.core import LoggerCallback, State


class MetricHistory(LoggerCallback):
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        
        self.history = []
        
        super().__init__()
        
    def on_batch_end(self, state: State):
        self.history.append(state.metric_manager.batch_values[self.metric_name])

