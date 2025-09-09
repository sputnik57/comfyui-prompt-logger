from .PromptLoggerUnified import PromptLoggerUnified
from .PromptLoggerUnified_2 import PromptLoggerUnified as PromptLoggerUnified_v2

NODE_CLASS_MAPPINGS = {
    "PromptLoggerUnified": PromptLoggerUnified,
    "PromptLoggerUnified_v2": PromptLoggerUnified_v2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptLoggerUnified": "Prompt Logger Unified",
    "PromptLoggerUnified_v2": "Prompt Logger Unified v2"
}
