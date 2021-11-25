_REGISTRY = {}


def get_item(category: str, item: str) -> object:
    _category = get_category(category)
    if item not in _category:
        raise ValueError(f"Item {item} not in category {category}")

    return _category[item]


def get_category(category: str) -> dict:
    if not category in _REGISTRY:
        raise ValueError(f"Category {category} not in registry")

    return _REGISTRY[category]


def register_item(category: str, item: object, name: str) -> None:
    _category = get_category(category)
    if item in _category:
        raise ValueError(f"Item {item} already in category {category}")

    _category[name] = item

    return item


def register_category(category: str) -> None:
    assert isinstance(category, str)

    if category in _REGISTRY:
        raise ValueError(f"Category {category} in registry already")

    _REGISTRY[category] = {}

    def get_func(item: str):
        return get_item(category, item)

    def register_func(obj: object, *, name: str = None):
        name = name if name is not None else obj.__name__.split(".")[-1]
        return register_item(category, obj, name)

    return get_func, register_func


get_model, register_model = register_category("model")
get_dataset, register_dataset = register_category("dataset")
