from typing import Any, Optional

from .base import LoggerCollection, LightningLoggerBase
from .csv_log import CSVLogger
from .wandb import WandbLogger

from hydra.core.singleton import Singleton


class Logger(metaclass=Singleton):
    def __init__(self) -> None:
        self.logger: Optional[LightningLoggerBase] = None

    def set_logger(self, logger: LightningLoggerBase):
        assert logger is not None
        assert isinstance(logger, LightningLoggerBase)
        self.logger = logger

    @staticmethod
    def get() -> LightningLoggerBase:
        instance = Logger.instance()
        if instance.logger is None:
            raise ValueError("Logger not set")
        return instance.logger

    @staticmethod
    def initialized() -> bool:
        instance = Logger.instance()
        return instance.logger is not None

    @staticmethod
    def instance(*args: Any, **kwargs: Any) -> "Logger":
        return Singleton.instance(Logger, *args, **kwargs)  # type: ignore
