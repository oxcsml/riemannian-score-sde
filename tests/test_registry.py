# %%
%load_ext autoreload
%autoreload 2

# %%
from score_sde.utils.registry import register_category, _REGISTRY
import functools
# %%
get_test, register_test = register_category("test")

# %%
register_test("string", name="test_string")

@register_test
def test_func():
    return 0

@register_test
class test_class:
    pass

# %%
