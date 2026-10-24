# Copyright (C) 2025 AIDC-AI
# Licensed under the MIT License.

import json
from typing import List, Dict, Any
from aiohttp import web
from ..utils.globals import LLM_DEFAULT_BASE_URL, OPENAI_API_KEY, OPENAI_BASE_URL, TENANT_ID, is_lmstudio_url
import server
import aiohttp
from ..utils.logger import log

MODELS_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)


async def fetch_models(base_url: str, api_key: str):
    """GET {base_url}/models. Returns (status, json_or_text)."""
    headers = {}
    # LMStudio usually runs without a key
    if api_key or not is_lmstudio_url(base_url):
        headers["Authorization"] = f"Bearer {api_key}"
    async with aiohttp.ClientSession(timeout=MODELS_REQUEST_TIMEOUT) as session:
        async with session.get(f"{base_url.rstrip('/')}/models", headers=headers) as response:
            if response.status == 200:
                return response.status, await response.json(content_type=None)
            return response.status, await response.text()


@server.PromptServer.instance.routes.get("/api/model_config")
async def list_models(request):
    """
    List available LLM models
    
    Returns:
        JSON response with models list in the format expected by frontend:
        {
            "models": [
                {"name": "model_name", "image_enable": boolean},
                ...
            ]
        }
    """
    try:
        log.info("Received list_models request")
        if TENANT_ID:
            model_list = [ "gemini-2.5-flash", "gpt-5-nano", "gpt-5-mini", "gpt-5" ]
            llm_config = []
            for model in model_list:
                llm_config.append({
                    "label": model,
                    "name": model,
                    "image_enable": True
                })
            return web.json_response({
                "models": llm_config
            })
        
        openai_api_key = request.headers.get('Openai-Api-Key') or OPENAI_API_KEY or ""
        openai_base_url = request.headers.get('Openai-Base-Url') or OPENAI_BASE_URL or LLM_DEFAULT_BASE_URL

        status, models = await fetch_models(openai_base_url, openai_api_key)
        llm_config = []
        if status == 200:
            for model in models.get('data', []):
                llm_config.append({
                    "label": model['id'],
                    "name": model['id'],
                    "image_enable": True
                })
        
        return web.json_response({
                "models": llm_config
            }
        )
        
    except Exception as e:
        log.error(f"Error in list_models: {str(e)}")
        return web.json_response({
            "error": f"Failed to list models: {str(e)}"
        }, status=500)


@server.PromptServer.instance.routes.get("/verify_openai_key")
async def verify_openai_key(req):
    """
    Verify if an OpenAI API key is valid by calling the OpenAI models endpoint
    Also supports LMStudio verification (which may not require an API key)
    
    Returns:
        JSON response with success status and message
    """
    openai_api_key = req.headers.get('Openai-Api-Key')
    openai_base_url = req.headers.get('Openai-Base-Url') or LLM_DEFAULT_BASE_URL
    is_lmstudio = is_lmstudio_url(openai_base_url)
    try:
        
        # For LMStudio, API key might not be required
        if not openai_api_key and not is_lmstudio:
            return web.json_response({
                "success": False, 
                "message": "No API key provided"
            })
        
        status, body = await fetch_models(openai_base_url, openai_api_key or "")

        if status == 200:
            success_message = "API key is valid" if not is_lmstudio else "LMStudio connection successful"
            return web.json_response({
                "success": True, 
                "data": True, 
                "message": success_message
            })
        else:
            log.error(f"API validation failed with status code: {status}")
            error_message = f"Invalid API key: HTTP {status} - {body}"
            if is_lmstudio:
                error_message = f"LMStudio connection failed: HTTP {status} - {body}"
            return web.json_response({
                "success": False, 
                "data": False,
                "message": error_message
            })
            
    except Exception as e:
        log.error(f"Error verifying API key/connection: {str(e)}")
        error_message = f"LMStudio connection error: {e}" if is_lmstudio else f"Invalid API key: {e}"
        return web.json_response({
            "success": False, 
            "data": False, 
            "message": error_message
        })