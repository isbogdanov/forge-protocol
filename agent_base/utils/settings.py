#from utils.llm_connector.connector.connector_settings import (
#    DEFAULT_PROVIDER,
#    DEFAULT_MODEL,
#)

import os

# Define the base directory of the agent (agent_base folder)
AGENT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PROVIDER = "openrouter"
# DEFAULT_MODEL = "qwen/qwen-plus-2025-07-28"
# DEFAULT_MODEL = "qwen/qwen-plus-2025-07-28"

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"

# DEFAULT_MODEL = "meta-llama/llama-4-maverick"

# DEFAULT_PROVIDER = "google"
# DEFAULT_MODEL = "google/gemini-2.5-flash-lite"

# DEFAULT_MODEL = "deepseek/deepseek-v3.2-exp"

# DEFAULT_MODEL = "openai/gpt-oss-20b"

# DEFAULT_MODEL = "meta-llama/llama-4-scout"

# DEFAULT_MODEL = "google/gemini-flash-1.5-8b"

# DEFAULT_MODEL = "mistralai/ministral-3b"

# DEFAULT_MODEL = "meta-llama/llama-4-maverick"

# DEFAULT_PROVIDER = "ollama"
# DEFAULT_MODEL = "llama3.1:8b"


PROVIDER = (DEFAULT_PROVIDER, DEFAULT_MODEL)


TOOLS_PROVIDER = PROVIDER


PROVIDER_PARAMETERS = {"temperature": 0, "max_tokens": 10000, "top_p": 0.7}

ANALYST_PROVIDER_PARAMETERS = {"temperature": 0, "max_tokens": 10000, "top_p": 0.7}

ACTION_CHOOSER_PARAMETERS = {"temperature": 0, "max_tokens": 10000, "top_p": 0.7}

# Configuration for the Reflector Agent - uses a powerful model for analysis
# REFLECTOR_PROVIDER = (DEFAULT_PROVIDER, "google/gemini-2.5-pro")

# REFLECTOR_PROVIDER = (DEFAULT_PROVIDER, "google/gemini-2.5-pro-preview")
# REFLECTOR_PROVIDER = (DEFAULT_PROVIDER, "google/gemini-2.5-flash-lite")

REFLECTOR_PROVIDER = (DEFAULT_PROVIDER, "google/gemini-2.5-flash-lite")

def update_provider_settings(provider_name, model_name):
    global PROVIDER, REFLECTOR_PROVIDER, TOOLS_PROVIDER, EXEMPLIFIER_PROVIDER
    new_provider = (provider_name, model_name)
    PROVIDER = new_provider
    TOOLS_PROVIDER = new_provider
    # For now, we also update Reflector/Exemplifier to use the same provider to ensure consistency
    # unless we want to keep them separate. The request was to override "provided and model".
    # Assuming the user wants to switch the main engine.
    # We generally want Reflector to be a strong model.
    # If the user switches to a weak model, Reflector might suffer.
    # However, if they switch to a different provider (e.g. OpenAI), we should probably switch Reflector too.
    REFLECTOR_PROVIDER = new_provider
    EXEMPLIFIER_PROVIDER = new_provider

def update_reflector_settings(provider_name, model_name):
    global REFLECTOR_PROVIDER
    REFLECTOR_PROVIDER = (provider_name, model_name)

def update_exemplifier_settings(provider_name, model_name):
    global EXEMPLIFIER_PROVIDER
    EXEMPLIFIER_PROVIDER = (provider_name, model_name)


REFLECTOR_PROVIDER_PARAMETERS = {"temperature": 0, "max_tokens": 20000, "top_p": 0.9}

# Configuration for the Exemplifier Agent - similar to Reflector
EXEMPLIFIER_PROVIDER = REFLECTOR_PROVIDER
EXEMPLIFIER_PROVIDER_PARAMETERS = {"temperature": 0, "max_tokens": 20000, "top_p": 0.9}

# Configuration for Online Learning
ONLINE_LEARNING_QUEUE_MAX_SIZE = 5

TOOLS_PROVIDER_PARAMETERS = {
    "temperature": 0,
    "max_tokens": 10000,
    "top_p": 0.7,
}

CONTEXT_LENGTH = 100000
