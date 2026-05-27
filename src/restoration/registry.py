from src.restoration.dcp import DCPDehazeFilter
from src.restoration.wmgf_derain import WMGFDerainFilter
from src.restoration.morph_guided_desnow import MorphGuidedDesnowFilter, MorphGuidedDesnowConfig
import yaml

def create_filter(filter_name, config_path=None):
    if filter_name.lower() == "dcp":
        return DCPDehazeFilter()
    elif filter_name.lower() == "wmgf":
        return WMGFDerainFilter()
    elif filter_name.lower() == "desnow":
        config = MorphGuidedDesnowConfig()
        if config_path:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    for k, v in config_data.items():
                        if hasattr(config, k):
                            setattr(config, k, v)
        return MorphGuidedDesnowFilter(config)
    else:
        raise ValueError(f"Unknown filter name: {filter_name}")
