import yaml
from utils.PTE.builder import PromptBuilder
import os

# Load the base agent definition
from utils.settings import AGENT_BASE_DIR
with open(os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/exemplifier/core.yaml"), "r") as f:
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

# Ensure common_knowledge is a list, defaulting to an empty list if it's None
if common_knowledge is None:
    common_knowledge = []

# Load agent-specific persistent knowledge
persistent_knowledge = []
persistent_knowledge_path = os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/exemplifier/persistent_knowledge.yaml")
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
reflection_knowledge_path = os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/exemplifier/reflection_knowledge.yaml")
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

# Load and merge examples only if the 'add_examples' flag is true
# Note: For the Exemplifier, these are "examples of how to create examples"
if agent_definition.get("add_examples", False):
    examples_path = os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/exemplifier/examples.yaml")
    if os.path.exists(examples_path):
        with open(examples_path, "r") as f:
            examples_definition = yaml.safe_load(f)
        
        if examples_definition and "examples" in examples_definition:
            agent_definition.setdefault("examples", []).extend(
                examples_definition["examples"]
            )

# The PromptBuilder needs a path, so we'll pass the original and then
# overwrite its internal template with our merged one.
_builder = PromptBuilder(
    agent_definition_path=os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/exemplifier/core.yaml")
)
_builder.agent_template = agent_definition


EXEMPLIFIER_SYSTEM_PROMPT = _builder.build()


def get_dynamic_exemplifier_system_prompt() -> str:
    # If the Exemplifier starts learning (populating its own reflection_knowledge),
    # we would need to reload it here similar to other agents.
    # For now, since it's a meta-agent, we can stick to the static build or reload if needed.
    
    # Reload logic (optional but good for consistency if we add meta-learning later)
    # Start with a fresh copy
    dynamic_agent_def = agent_definition.copy()
    
    current_agent_specific_knowledge = []
    if os.path.exists(reflection_knowledge_path):
        with open(reflection_knowledge_path, "r") as f:
            specific_knowledge_data = yaml.safe_load(f)
            if specific_knowledge_data and "reflection_knowledge" in specific_knowledge_data:
                current_agent_specific_knowledge = specific_knowledge_data.get("reflection_knowledge", []) or []

    full_knowledge = common_knowledge + persistent_knowledge + current_agent_specific_knowledge
    if full_knowledge:
        dynamic_agent_def["reflection_knowledge"] = full_knowledge

    _builder.agent_template = dynamic_agent_def
    prompt = _builder.build()
    _builder.agent_template = agent_definition

    return prompt
