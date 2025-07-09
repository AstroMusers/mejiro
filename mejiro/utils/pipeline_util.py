import importlib


def initialize_instrument_class(instrument_name):
    base_module_path = "mejiro.instruments"
    class_map = {
        "hwo": "HWO",
        "roman": "Roman"
    }

    if instrument_name.lower() not in class_map:
        raise ValueError(f"Unknown instrument: {instrument_name}")

    module_path = f"{base_module_path}.{instrument_name.lower()}"
    module = importlib.import_module(module_path)
    class_name = class_map[instrument_name.lower()]
    cls = getattr(module, class_name)
    instance = cls()
    
    return instance
