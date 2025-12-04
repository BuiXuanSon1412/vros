# ui/config/__init__.py
"""Configuration UI components - Problem-specific modules"""

from .system_config_p1 import render_system_config_p1
from .system_config_p2 import render_system_config_p2
from .system_config_p3 import render_system_config_p3
from .algorithm_config_p1 import render_algorithm_config_p1
from .algorithm_config_p2 import render_algorithm_config_p2
from .algorithm_config_p3 import render_algorithm_config_p3
from .dataset_display import render_dataset_info

__all__ = [
    "render_system_config_p1",
    "render_system_config_p2",
    "render_system_config_p3",
    "render_algorithm_config_p1",
    "render_algorithm_config_p2",
    "render_algorithm_config_p3",
    "render_dataset_info",
]
