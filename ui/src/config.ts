// Copyright (C) 2025 AIDC-AI
// Licensed under the MIT License.

const isDevelopment = import.meta.env.MODE === 'development'

const defaultApiBaseUrl = 'http://localhost:8000'

export const github_url = 'https://github.com/AIDC-AI/ComfyUI-Copilot'

const remoteApiBaseUrl: string | undefined = import.meta.env.VITE_API_BASE_URL

export const config = {
  apiBaseUrl: isDevelopment 
    ? defaultApiBaseUrl 
    : (remoteApiBaseUrl || defaultApiBaseUrl),
  // The hosted Copilot server has been suspended. Features that only it provides (account signup,
  // usage tracking, announcements, the workflow library, AI prompt writing via the server) are
  // only used by builds that set VITE_API_BASE_URL to a compatible server.
  remoteServerEnabled: Boolean(remoteApiBaseUrl),
}
