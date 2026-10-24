from agents._config import set_default_openai_api
from agents.tracing import set_tracing_disabled

from .utils.globals import ENABLE_TRACING

set_default_openai_api("chat_completions")
set_tracing_disabled(not ENABLE_TRACING)
