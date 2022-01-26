# %%
%load_ext autoreload
%autoreload 2

# %%
from score_sde.utils.loggers_pl import Logger
from score_sde.utils.loggers_pl import CSVLogger
# %%
Logger.instance()

# %%

Logger.instance().set_logger(CSVLogger('.'))

# %%

Logger.get()
# %%
