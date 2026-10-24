// Copyright (C) 2025 AIDC-AI
// Licensed under the MIT License.

import { api } from "./utils/comfyapp";

export function fetchApi(
  route: string,
  options?: RequestInit,
): Promise<Response> {
  if (api == null) {
    console.error("api is null!");
    throw new Error("api is null!");
  }
  return api.fetchApi(route, options);
}
