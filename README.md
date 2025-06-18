# Oncology Backend - LLM Integration Journey

## Overview
This repository contains our journey in integrating Large Language Models (LLMs) into our oncology backend system. We've implemented support for multiple LLM providers, with a focus on LiteLLM integration.

## Current Status
- ✅ Code implementation is complete and follows best practices
- ✅ Environment configuration is properly set up
- ❌ LiteLLM proxy server (rillavoice) configuration issue identified

## Implementation Details

### Supported LLM Providers
1. **LiteLLM** (Primary)
   - Base URL: https://litellm.rillavoice.com/v1
   - Configuration in `.env`:
     ```
     LITELLM_BASE_URL=https://litellm.rillavoice.com/v1
     LITELLM_API_KEY=sk-rilla-vibes
     LLM_PROVIDER=LITELLM
     ```

2. **Google Gemini** (Fallback)
   - Configuration in `.env`:
     ```
     GOOGLE_API_KEY=your_key_here
     ```

### Key Files
- `backend/core/llm_config.py`: Configuration management
- `backend/core/llm_utils.py`: LLM interaction utilities
- `test_llm.py`: Integration testing script

## Known Issues

### LiteLLM Proxy Server Issue
We've identified a configuration issue with the rillavoice LiteLLM proxy server:

1. **Error Pattern**:
   ```
   litellm.NotFoundError: AnthropicException - {"detail":"Not Found"}
   ```

2. **What This Means**:
   - Our code successfully connects to the proxy
   - The proxy forwards requests to Anthropic
   - Anthropic rejects the model names as invalid

3. **Tested Models**:
   - `anthropic/claude-3-5-sonnet`
   - `anthropic/claude-3-5-haiku`
   - `anthropic/claude-3-opus`

4. **Root Cause**:
   The proxy server appears to be misconfigured for Anthropic model access. This is a server-side issue that needs to be addressed by the rillavoice service administrators.

## Next Steps
1. Report the issue to rillavoice service administrators
2. Request proper configuration of Anthropic model access
3. Consider alternative LLM providers if needed

## Testing
Run the test script to verify LLM functionality:
```bash
python test_llm.py
```

## Environment Setup
1. Copy `.env.example` to `.env`
2. Configure your API keys and settings
3. Ensure no quotes in environment variable values
4. Set `LLM_PROVIDER` to either "LITELLM" or "GEMINI"

## Contributing
Please document any new findings or solutions in this README. 