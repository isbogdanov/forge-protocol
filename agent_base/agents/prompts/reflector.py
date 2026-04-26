import yaml
from utils.PTE.builder import PromptBuilder
import os
import re

# Load the base agent definition
from utils.settings import AGENT_BASE_DIR
with open(os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/reflector/core.yaml"), "r") as f:
    agent_definition = yaml.safe_load(f)

# Load common knowledge if it exists
common_knowledge = []
common_knowledge_path = os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/common_knowledge.yaml")
if os.path.exists(common_knowledge_path):
    with open(common_knowledge_path, "r") as f:
        common_knowledge_data = yaml.safe_load(f)
        common_knowledge = (
            common_knowledge_data.get("reflection_knowledge", [])
            if common_knowledge_data
            else []
        )

if common_knowledge is None:
    common_knowledge = []

# Load agent-specific persistent knowledge
persistent_knowledge = []
persistent_knowledge_path = os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/reflector/persistent_knowledge.yaml")
if os.path.exists(persistent_knowledge_path):
    with open(persistent_knowledge_path, "r") as f:
        persistent_knowledge_data = yaml.safe_load(f)
        if (
            persistent_knowledge_data
            and "reflection_knowledge" in persistent_knowledge_data
        ):
            persistent_knowledge = (
                persistent_knowledge_data.get("reflection_knowledge", []) or []
            )

# Load and merge agent-specific reflection knowledge if it exists
agent_specific_knowledge = []
reflection_knowledge_path = os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/reflector/reflection_knowledge.yaml")
if os.path.exists(reflection_knowledge_path):
    with open(reflection_knowledge_path, "r") as f:
        specific_knowledge_data = yaml.safe_load(f)
        if (
            specific_knowledge_data
            and "reflection_knowledge" in specific_knowledge_data
        ):
            agent_specific_knowledge = (
                specific_knowledge_data.get("reflection_knowledge", []) or []
            )

# Combine knowledge and update the definition
combined_knowledge = common_knowledge + persistent_knowledge + agent_specific_knowledge
if combined_knowledge:
    agent_definition["reflection_knowledge"] = combined_knowledge

_builder = PromptBuilder(
    agent_definition_path=os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/reflector/core.yaml")
)
_builder.agent_template = agent_definition

REFLECTOR_SYSTEM_PROMPT = _builder.build()


def get_dynamic_reflector_system_prompt(
    reward_threshold: float, max_reflection_rules: int
) -> str:
    # Use targeted regex substitutions to safely replace only the desired placeholders,
    # avoiding errors with other curly braces in the prompt.
    prompt = re.sub(
        r"\{reward_threshold\}", str(reward_threshold), REFLECTOR_SYSTEM_PROMPT
    )
    prompt = re.sub(r"\{max_reflection_rules\}", str(max_reflection_rules), prompt)
    return prompt
