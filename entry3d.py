from pathlib import Path

from sacred import Experiment
from tensorflow.keras.callbacks import ModelCheckpoint

from config import setup, initialize
from models.DINs import DINs
from core.trainer import Trainer, Evaluator3D, Evaluator3DWithBox
from data_kits import datasets

ex = setup(
    Experiment("DINs", base_dir=Path(__file__).parent, save_git_info=False)
)


@ex.command
def train(_run, _config):
    opt, logger = initialize(ex, _run, _config)
    logger.info("Initialize ==> Training")

    # Build Model and Trainer
    model = DINs(opt, logger)
    trainer = Trainer(opt, logger, model)

    # Model Checkpoint
    trainer.callbacks.extend([
        ModelCheckpoint(Path(_run.logdir_) / "ckpt/ckpt",
                        verbose=1,
                        save_weights_only=True),
        ModelCheckpoint(Path(_run.logdir_) / "bestckpt/bestckpt",
                        verbose=1,
                        monitor="dice",
                        save_weights_only=True,
                        save_best_only=True,
                        mode='max')
    ])

    # Build data loader
    train_dataset = datasets.loads(opt, logger, "train")
    eval_dataset = datasets.loads(opt, logger, "eval_online")

    # Start training
    trainer.start_training_loop(train_dataset, eval_dataset)

    logger.info(f"============ Training finished - id {_run._id} ============")


@ex.command
def test(_run, _config):
    opt, logger = initialize(ex, _run, _config)
    logger.info("Initialize ==> Testing")

    # Build Model and Evaluator
    model = DINs(opt, logger)

    if opt.use_box:
        evaluator = Evaluator3DWithBox(opt, logger, model, _run)
    else:
        evaluator = Evaluator3D(opt, logger, model, _run)

    # Build data loader
    eval_dataset = datasets.loads(opt, logger, opt.test_set)

    # Start evaluating
    evaluator.start_evaluating_loop(eval_dataset)


if __name__ == "__main__":
    ex.run_commandline()
