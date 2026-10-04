"""Callbacks used for every training run."""

import time

import keras


class EpochTimer(keras.callbacks.Callback):
    """Adds the epoch duration (seconds) to the logs, so CSVLogger saves it."""

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["epoch_time"] = time.perf_counter() - self._start


class RestoreBest(keras.callbacks.Callback):
    """When a run is resumed, give EarlyStopping and ReduceLROnPlateau back the best val_loss reached
    before the interruption (their on_train_begin resets it)."""

    def __init__(self, callbacks, best):
        super().__init__()
        self.callbacks, self.best = callbacks, best

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.best = self.best


def make_callbacks(
    run_dir,
    checkpoint_path,
    monitor="val_loss",
    early_stopping_patience=8,
    reduce_lr_patience=3,
    reduce_lr_factor=0.3,
    min_lr=1e-6,
    log_name="history.csv",
    resume_best=None,
):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=monitor, patience=early_stopping_patience, restore_best_weights=True, verbose=1
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor, factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=min_lr, verbose=1
    )
    callbacks = [
        EpochTimer(),  # before CSVLogger, otherwise epoch_time is not logged
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor=monitor, save_best_only=True, initial_value_threshold=resume_best
        ),
        early_stopping,
        reduce_lr,
        keras.callbacks.CSVLogger(str(run_dir / log_name), append=resume_best is not None),
    ]
    if resume_best is not None:
        callbacks.append(RestoreBest([early_stopping, reduce_lr], resume_best))  # must come after them
    return callbacks
