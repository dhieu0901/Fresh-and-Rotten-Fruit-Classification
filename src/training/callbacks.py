"""Callbacks used by every training run (same settings for all three models)."""
import time

import keras


class EpochTimer(keras.callbacks.Callback):
    """Adds `epoch_time` (seconds) to the epoch logs (picked up by CSVLogger; Keras logs `learning_rate` itself)."""

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["epoch_time"] = time.perf_counter() - self._start


class RestoreBest(keras.callbacks.Callback):
    """Resumed run: give EarlyStopping / ReduceLROnPlateau back the best value reached before the
    interruption (their on_train_begin resets it). Must come after them in the callback list."""

    def __init__(self, callbacks, best):
        super().__init__()
        self.callbacks, self.best = callbacks, best

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.best = self.best


def make_callbacks(run_dir, checkpoint_path, monitor="val_loss", early_stopping_patience=8,
                   reduce_lr_patience=3, reduce_lr_factor=0.3, min_lr=1e-6, log_name="history.csv", resume_best=None):
    """Best-checkpoint saving, early stopping, LR reduction on plateau, per-epoch CSV log.

    resume_best: best monitored value of an interrupted run that continues from its checkpoint
    (scripts/train.py --resume) - the checkpoint is then only replaced by a better epoch and the
    CSV log is appended to.
    """
    early_stopping = keras.callbacks.EarlyStopping(monitor=monitor, patience=early_stopping_patience,
                                                   restore_best_weights=True, verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=reduce_lr_factor, patience=reduce_lr_patience,
                                                  min_lr=min_lr, verbose=1)
    callbacks = [
        EpochTimer(),  # must run before CSVLogger so its values are logged
        keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor=monitor, save_best_only=True,
                                        initial_value_threshold=resume_best),
        early_stopping,
        reduce_lr,
        keras.callbacks.CSVLogger(str(run_dir / log_name), append=resume_best is not None),
    ]
    if resume_best is not None:
        callbacks.append(RestoreBest([early_stopping, reduce_lr], resume_best))
    return callbacks
