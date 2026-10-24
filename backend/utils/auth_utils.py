# Copyright (C) 2025 AIDC-AI
# Licensed under the MIT License.

"""
Authentication utilities for ComfyUI Copilot
"""

from typing import Optional


def extract_api_key(request) -> Optional[str]:
    """
    Extract the Bearer token from the Authorization header.

    The key is returned rather than stored globally; callers put it into the
    request-scoped config (``config["copilot_api_key"]``) so concurrent
    requests never see each other's credentials.
    """
    auth_header = request.headers.get('Authorization') or ''
    if auth_header.startswith('Bearer '):
        api_key = auth_header[7:].strip()
        if api_key and api_key not in ('undefined', 'null'):
            return api_key
    return None
