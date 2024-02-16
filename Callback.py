from stable_baselines3.common.callbacks import *
from stable_baselines3.common.logger    import HParam
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.callbacks import BaseCallback


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate(self.model._current_progress_remaining),
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True
    
    
class LogRewardCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    
    
    def _on_step(self) -> bool:
        print(type(self.locals["self"].env))
        return True
    
    





class SummaryWriterCallback(BaseCallback):

    def _on_training_start(self):
        self._log_freq = 1000  # log every 1000 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:
            # You can have access to info from the env using self.locals.
            # for instance, when using one env (index 0 of locals["infos"]):
            # lap_count = self.locals["infos"][0]["lap_count"]
            # self.tb_formatter.writer.add_scalar("train/lap_count", lap_count, self.num_timesteps)

            self.tb_formatter.writer.add_text("direct_access", "this is a value", self.num_timesteps)
            self.tb_formatter.writer.flush()


