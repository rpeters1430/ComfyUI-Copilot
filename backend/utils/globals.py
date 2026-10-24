'''
Author: ai-business-hql qingli.hql@alibaba-inc.com
Date: 2025-08-08 17:14:52
LastEditors: ai-business-hql ai.bussiness.hql@gmail.com
LastEditTime: 2025-12-15 14:49:18
FilePath: /comfyui_copilot/backend/utils/globals.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

"""
Global utilities for managing application-wide state and configuration.
"""

import contextvars
import os
import threading
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

class GlobalState:
    """Thread-safe global state manager for application-wide configuration."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {
            'LANGUAGE': 'en',  # Default language
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a global state value."""
        with self._lock:
            return self._state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a global state value."""
        with self._lock:
            self._state[key] = value
    
    def get_language(self) -> str:
        """Get the current language setting."""
        return self.get('LANGUAGE', 'en')
    
    def set_language(self, language: str) -> None:
        """Set the current language setting."""
        self.set('LANGUAGE', language)
    
    def update(self, **kwargs) -> None:
        """Update multiple state values at once."""
        with self._lock:
            self._state.update(kwargs)
    
    def get_all(self) -> Dict[str, Any]:
        """Get a copy of all global state."""
        with self._lock:
            return self._state.copy()

# Global instance
_global_state = GlobalState()

# Convenience functions for external access
def get_global(key: str, default: Any = None) -> Any:
    """Get a global state value."""
    return _global_state.get(key, default)

def set_global(key: str, value: Any) -> None:
    """Set a global state value."""
    _global_state.set(key, value)

# Request-scoped so concurrent requests in different languages don't overwrite each other
_request_language: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('language', default=None)

def get_language() -> str:
    """Get the language for the current request (falls back to the last one set)."""
    return _request_language.get() or _global_state.get_language() or 'en'

def set_language(language: str) -> None:
    """Set the language for the current request."""
    _request_language.set(language)
    _global_state.set_language(language)

def update_globals(**kwargs) -> None:
    """Update multiple global values at once."""
    _global_state.update(**kwargs)

def get_all_globals() -> Dict[str, Any]:
    """Get a copy of all global state."""
    return _global_state.get_all()

def get_comfyui_copilot_api_key() -> Optional[str]:
    """Get the ComfyUI Copilot API key for the current request.

    The key is request-scoped (stored in the request context config) so that
    concurrent users never share each other's credentials. The process-wide
    value is only a fallback for code running outside a request.
    """
    from .request_context import get_config
    config = get_config() or {}
    return config.get('copilot_api_key') or _global_state.get('comfyui_copilot_api_key')

def set_comfyui_copilot_api_key(api_key: str) -> None:
    """Set the ComfyUI Copilot API key."""
    _global_state.set('comfyui_copilot_api_key', api_key)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# The upstream hosted Copilot server has been suspended, so nothing defaults to it any more.
# Set BACKEND_BASE_URL (or COPILOT_REMOTE_MCP_URL) only if you run a compatible server yourself.
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL") or None
REMOTE_MCP_URL = os.getenv("COPILOT_REMOTE_MCP_URL") or (
    f"{BACKEND_BASE_URL.rstrip('/')}/mcp-server/mcp" if BACKEND_BASE_URL else None
)
# Optional web-search MCP server (SSE). Disabled unless configured.
SEARCH_MCP_URL = os.getenv("COPILOT_SEARCH_MCP_URL") or None
SEARCH_MCP_API_KEY = os.getenv("COPILOT_SEARCH_MCP_API_KEY") or None
MCP_CONNECT_TIMEOUT = float(os.getenv("COPILOT_MCP_CONNECT_TIMEOUT", "15"))

LMSTUDIO_DEFAULT_BASE_URL = "http://localhost:1234/v1"
LLM_DEFAULT_BASE_URL = os.getenv("CC_DEFAULT_BASE_URL") or "https://api.openai.com/v1"
# Chat model used when the request does not select one.
DEFAULT_CHAT_MODEL = os.getenv("CC_DEFAULT_MODEL") or None
# Model for workflow agents (rewrite/debug). When unset, they reuse the chat model.
WORKFLOW_MODEL_NAME = os.getenv("WORKFLOW_MODEL_NAME") or None

# LLM-related env defaults (used as fallback when request config does not provide values)
OPENAI_API_KEY = os.getenv("CC_OPENAI_API_KEY") or None
OPENAI_BASE_URL = os.getenv("CC_OPENAI_BASE_URL") or None
WORKFLOW_LLM_API_KEY = os.getenv("WORKFLOW_LLM_API_KEY") or None
WORKFLOW_LLM_BASE_URL = os.getenv("WORKFLOW_LLM_BASE_URL") or None
WORKFLOW_LLM_MODEL = os.getenv("WORKFLOW_LLM_MODEL") or WORKFLOW_MODEL_NAME
DISABLE_WORKFLOW_GEN = _env_flag("DISABLE_WORKFLOW_GEN")
VISION_LLM_MODEL = os.getenv("VISION_LLM_MODEL") or None
VISION_LLM_BASE_URL = os.getenv("VISION_LLM_BASE_URL") or None
VISION_LLM_API_KEY = os.getenv("VISION_LLM_API_KEY") or None
TOOL_MODE = (os.getenv("COPILOT_TOOL_MODE") or "auto").strip().lower()
# "max_tokens" or "max_completion_tokens"; unset = picked per endpoint/model
TOKEN_LIMIT_PARAM = (os.getenv("CC_TOKEN_LIMIT_PARAM") or "").strip() or None
# openai-agents tracing uploads run traces (prompts, workflows) to OpenAI. Opt-in only.
ENABLE_TRACING = _env_flag("COPILOT_ENABLE_TRACING")
# Extra folders of workflow JSON files for the local workflow library (os.pathsep-separated)
WORKFLOW_DIRS = [d.strip() for d in (os.getenv("COPILOT_WORKFLOW_DIRS") or "").split(os.pathsep) if d.strip()]

TENANT_ID = os.getenv("TENANT_ID") or None

def apply_llm_env_defaults(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Apply LLM-related defaults with precedence:
    request config > .env > hard-coded defaults.

    This function does NOT mutate the incoming config.
    """
    cfg: Dict[str, Any] = dict(config or {})

    # Chat LLM (OpenAI-compatible) settings
    if not cfg.get("openai_api_key") and OPENAI_API_KEY:
        cfg["openai_api_key"] = OPENAI_API_KEY
    if not cfg.get("openai_base_url") and OPENAI_BASE_URL:
        cfg["openai_base_url"] = OPENAI_BASE_URL

    # Workflow LLM settings (tools/agents that might use a different LLM)
    if not cfg.get("workflow_llm_api_key") and WORKFLOW_LLM_API_KEY:
        cfg["workflow_llm_api_key"] = WORKFLOW_LLM_API_KEY
    if not cfg.get("workflow_llm_base_url") and WORKFLOW_LLM_BASE_URL:
        cfg["workflow_llm_base_url"] = WORKFLOW_LLM_BASE_URL
    if not cfg.get("workflow_llm_model") and WORKFLOW_LLM_MODEL:
        cfg["workflow_llm_model"] = WORKFLOW_LLM_MODEL

    # Vision model used to describe images for chat models that can't see them
    if not cfg.get("vision_llm_model") and VISION_LLM_MODEL:
        cfg["vision_llm_model"] = VISION_LLM_MODEL
    if not cfg.get("vision_llm_base_url") and VISION_LLM_BASE_URL:
        cfg["vision_llm_base_url"] = VISION_LLM_BASE_URL
    if not cfg.get("vision_llm_api_key") and VISION_LLM_API_KEY:
        cfg["vision_llm_api_key"] = VISION_LLM_API_KEY
    if not cfg.get("tool_mode"):
        cfg["tool_mode"] = TOOL_MODE

    return cfg


def is_lmstudio_url(base_url: str) -> bool:
    """Check if the base URL is likely LMStudio based on common patterns."""
    if not base_url:
        return False

    base_url_lower = base_url.lower()
    # Common LMStudio patterns (supporting various ports and configurations)
    lmstudio_patterns = [
        "localhost:1234",        # Standard LMStudio port
        "127.0.0.1:1234",
        "0.0.0.0:1234",
        ":1234/v1",
        "localhost:1235",        # Alternative port some users might use
        "127.0.0.1:1235",
        "0.0.0.0:1235",
        ":1235/v1",
        "localhost/v1",          # Generic localhost patterns
        "127.0.0.1/v1"
    ]

    return any(pattern in base_url_lower for pattern in lmstudio_patterns)
