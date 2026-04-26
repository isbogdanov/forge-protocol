import logging
from typing import Dict, Any, List
import os
import json
import yaml

from base_classes.react_agent_base import ReActAgent
from logs.config.log_config import add_dynamic_logger
from agents.prompts.reflector import get_dynamic_reflector_system_prompt
from agents.prompts.exemplifier import get_dynamic_exemplifier_system_prompt
from agents.reflector_tool_executor import ReflectorToolExecutor
from agents.exemplifier_tool_executor import ExemplifierToolExecutor
import utils.settings as settings
from utils.settings import (
    # REFLECTOR_PROVIDER, # Access via settings.REFLECTOR_PROVIDER
    REFLECTOR_PROVIDER_PARAMETERS,
    # EXEMPLIFIER_PROVIDER, # Access via settings.EXEMPLIFIER_PROVIDER
    EXEMPLIFIER_PROVIDER_PARAMETERS
)


def _clean_heuristic(raw_heuristic: str) -> str:
    if not isinstance(raw_heuristic, str):
        return "Error: Invalid heuristic format."

    if "Answer:" in raw_heuristic:
        answer_pos = raw_heuristic.find("Answer:")
        raw_heuristic = raw_heuristic[answer_pos + len("Answer:") :].strip()

    raw_heuristic = raw_heuristic.strip()
    if raw_heuristic.startswith("Thought:"):
        raw_heuristic = raw_heuristic[len("Thought:") :].strip()

    raw_heuristic = raw_heuristic.replace("PAUSE", "").strip()

    # Remove markdown code blocks if present
    if raw_heuristic.startswith("```json"):
        raw_heuristic = raw_heuristic[7:]
    elif raw_heuristic.startswith("```"):
        raw_heuristic = raw_heuristic[3:]

    if raw_heuristic.endswith("```"):
        raw_heuristic = raw_heuristic[:-3]

    return raw_heuristic.strip()


def _prune_trajectory_user_history(trajectories: Dict[str, Any]) -> Dict[str, Any]:
    history_marker = "\n\n\nStep History:"
    placeholder = "\n\n\n[... same as above ...]\n\n\n"

    for traj_name, traj_content in trajectories.items():
        if "steps" not in traj_content:
            continue
        for step_key, step_data in traj_content["steps"].items():
            if "messages" not in step_data:
                continue
            for message in step_data["messages"]:
                if message.get("role") == "user" and history_marker in message.get(
                    "content", ""
                ):
                    parts = message["content"].split(history_marker, 1)
                    # The second part contains the history and the actual question.
                    # We need to find where the history JSON ends.
                    try:
                        # Find the closing brace of the JSON history block
                        end_of_history = parts[1].rindex("}") + 1
                        actual_question = parts[1][end_of_history:].strip()
                        message["content"] = (
                            f"{parts[0].strip()}{placeholder}{actual_question}"
                        )
                    except ValueError:
                        # Fallback if parsing fails, just in case
                        message["content"] = f"{parts[0].strip()}{placeholder}"

    return trajectories


def run_unified_reflection_task(
    snapshot: Dict[str, Any], output_dir: str, current_attempt: int
) -> Dict[str, Dict[str, list]]:

    # Extract lists of agent roles from the snapshot
    agents_to_improve = snapshot.get("agents_to_improve", [])
    helper_agents = snapshot.get("helper_agents", [])
    all_agent_names = set(agents_to_improve + helper_agents)

    # Work with a copy to avoid mutating the original dict in a multi-threaded context
    all_trajectories = snapshot["all_trajectories"].copy()

    # Prune redundant user history from all trajectories unless disabled
    if not snapshot.get("disable_trajectory_pruning", False):
        all_trajectories = _prune_trajectory_user_history(all_trajectories)

    # Conditionally prune system prompts from helper agents, but never from agents to improve
    if not snapshot.get("disable_system_prompt_pruning", False):
        for agent_name in helper_agents:
            if (
                agent_name in all_trajectories
                and "system_prompt" in all_trajectories[agent_name]
            ):
                del all_trajectories[agent_name]["system_prompt"]

    # --- Prompt Construction ---
    knowledge_section_str = "<KNOWLEDGE_BASE>\n"
    for agent_name in all_agent_names:
        knowledge = snapshot.get("knowledge", {}).get(agent_name, {})
        persistent = knowledge.get("persistent", [])
        reflection = knowledge.get("reflection", [])
        tactical = knowledge.get("tactical", [])

        knowledge_section_str += f"  <AGENT_KNOWLEDGE name='{agent_name}'>\n"

        persistent_str = (
            "\n".join(f"    - {item}" for item in persistent)
            if persistent
            else "    - None"
        )
        knowledge_section_str += f"    <PERSISTENT_KNOWLEDGE type='READ-ONLY'>\n{persistent_str}\n    </PERSISTENT_KNOWLEDGE>\n"

        editable_knowledge = reflection + tactical
        if editable_knowledge:
            reflection_str = "\n".join(
                f"    {i+1}. {item}" for i, item in enumerate(editable_knowledge)
            )
        else:
            reflection_str = "    - None"
        knowledge_section_str += (
            f"    <EDITABLE_KNOWLEDGE>\n{reflection_str}\n    </EDITABLE_KNOWLEDGE>\n"
        )
        knowledge_section_str += f"  </AGENT_KNOWLEDGE>\n"
    knowledge_section_str += "</KNOWLEDGE_BASE>"

    # --- Build non-XML current_rules_list (for simpler prompt format) ---
    current_rules_list = ""
    for agent_name in agents_to_improve:
        knowledge = snapshot.get("knowledge", {}).get(agent_name, {})
        persistent = knowledge.get("persistent", [])
        reflection = knowledge.get("reflection", [])
        tactical = knowledge.get("tactical", [])
        
        current_rules_list += f"AGENT: {agent_name}\n"
        current_rules_list += "[READ-ONLY]\n"
        if persistent:
            current_rules_list += "\n".join(f"- {item}" for item in persistent) + "\n"
        else:
            current_rules_list += "- None\n"
        
        current_rules_list += "\n[EDITABLE - use these indices for deletion]\n"
        editable_knowledge = reflection + tactical
        if editable_knowledge:
            current_rules_list += "\n".join(f"{i+1}. {item}" for i, item in enumerate(editable_knowledge)) + "\n"
        else:
            current_rules_list += "- None\n"
        current_rules_list += "\n"

    trajectories_section_str = "<TRAJECTORIES>\n"
    for agent_name in all_agent_names:
        trajectory = all_trajectories.get(agent_name, {})
        trajectories_section_str += f"  <AGENT_TRAJECTORY name='{agent_name}'>\n"
        trajectories_section_str += f"    {json.dumps(trajectory, indent=2)}\n"
        trajectories_section_str += f"  </AGENT_TRAJECTORY>\n"
    trajectories_section_str += "</TRAJECTORIES>"

    # Format the learning history for the prompt
    learning_history = snapshot.get("session_learning_history", [])
    learning_history_str = (
        yaml.dump(learning_history, indent=2)
        if learning_history
        else "First learning attempt, no history yet."
    )

    with open(os.path.join(settings.AGENT_BASE_DIR, "agents/prompts/definitions/reflector/initial_prompt.yaml"), "r") as f:
        prompt_data = yaml.safe_load(f)
        if "prompt" in prompt_data and isinstance(prompt_data["prompt"], str):
             prompt_template = prompt_data["prompt"]
        else:
             # Fallback/Support for opening/closing structure like Exemplifier
             prompt_template = prompt_data.get("prompt", {}).get("opening", "") + "\n" + prompt_data.get("prompt", {}).get("closing", "")

    # Construct the final dynamic prompt
    agents_to_improve_str = "\n".join(f"- {name}" for name in agents_to_improve)
    helper_agents_str = (
        "\n".join(f"- {name}" for name in helper_agents) if helper_agents else "- None"
    )

    full_prompt = prompt_template.format(
        failed_step=snapshot["failed_step"],
        reward=snapshot["reward"],
        total_reward=snapshot.get("total_reward", "N/A"),
        reward_threshold=snapshot.get("reward_threshold", -0.1),
        max_reflection_rules=snapshot.get("max_reflection_rules", 100),
        learning_history=learning_history_str,
        agents_to_improve=agents_to_improve_str,
        context_agents=helper_agents_str,
        knowledge_base=knowledge_section_str,
        current_rules_list=current_rules_list,
        trajectories=trajectories_section_str,
    )

    # --- Dynamic Logging & Agent Setup ---
    logger = logging.getLogger(__name__)
    logger_name = "Reflector.Unified"
    log_file_path = os.path.join(output_dir, f"{logger_name}.log")
    add_dynamic_logger(logger_name, log_file_path)
    task_logger = logging.getLogger(logger_name)

    trajectories_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(trajectories_dir, exist_ok=True)
    reflector_log_path = os.path.join(trajectories_dir, "Reflector_Unified.json")

    reflector_executor = ReflectorToolExecutor(logger=task_logger)

    dynamic_system_prompt = get_dynamic_reflector_system_prompt(
        reward_threshold=snapshot.get("reward_threshold", -0.1),
        max_reflection_rules=snapshot.get("max_reflection_rules", 100),
    )

    reflector_agent = ReActAgent(
        system_prompt=dynamic_system_prompt,
        llm_provider=settings.REFLECTOR_PROVIDER,
        llm_provider_parameters=REFLECTOR_PROVIDER_PARAMETERS,
        logger=task_logger,
        log_name=logger_name,
        episode_step=current_attempt,
        trajectory_log_path=reflector_log_path,
    )

    # --- Self-Correction Loop ---
    max_retries = 5
    final_json_output = None
    next_prompt = full_prompt

    for i in range(max_retries):
        raw_output = reflector_agent.run(next_prompt, reflector_executor)
        cleaned_output = _clean_heuristic(raw_output)

        try:
            parsed_json = json.loads(cleaned_output)
            # Validate the structure for the unified format
            for agent_name in agents_to_improve:
                if agent_name not in parsed_json:
                    raise ValueError(
                        f"Validation failed: Missing key for agent '{agent_name}' in JSON output."
                    )
                if (
                    "delete" not in parsed_json[agent_name]
                    or "add" not in parsed_json[agent_name]
                ):
                    raise ValueError(
                        f"Validation failed: Missing 'delete' or 'add' key for agent '{agent_name}'."
                    )
                if "rationale" not in parsed_json[agent_name]:
                    raise ValueError(
                        f"Validation failed: Missing 'rationale' key for agent '{agent_name}'."
                    )

            final_json_output = parsed_json
            task_logger.info(
                f"Successfully received valid unified JSON on attempt {i+1}."
            )
            break
        except (json.JSONDecodeError, ValueError) as e:
            task_logger.warning(
                f"Attempt {i+1}/{max_retries} failed to produce valid unified JSON. Error: {e}"
            )
            next_prompt = (
                f"CRITICAL: Your previous response was NOT valid JSON. The error was: {e}.\n\n"
                f"RULES YOU MUST FOLLOW:\n"
                f"1. NO MARKDOWN - Do not wrap JSON in ```json blocks\n"
                f"2. NO COMMENTS - JSON does not support comments (// or /* */)\n"
                f"3. Output ONLY the raw JSON object, nothing else\n\n"
                f"Provide a single, valid JSON object with a key for each agent in <AGENTS_TO_IMPROVE>.\n"
                f"Your invalid response was:\n{cleaned_output}"
            )

    if not final_json_output:
        task_logger.error(
            f"Failed to get valid JSON after {max_retries} attempts. Defaulting to empty operations for all agents."
        )
        # Create a default empty structure for all agents to improve
        final_json_output = {
            agent_name: {
                "rationale": "Reflector failed to produce valid output after retries.",
                "delete": [],
                "add": [],
            }
            for agent_name in agents_to_improve
        }

    task_logger.info(
        f"Generated unified heuristics: {json.dumps(final_json_output, indent=2)}"
    )
    return final_json_output


def run_example_generation_task(
    snapshot: Dict[str, Any], output_dir: str, current_attempt: int
) -> Dict[str, Dict[str, Any]]:
    """
    Runs the Exemplifier agent to generate Gold Standard examples from failed trajectories.
    """
    # Extract lists of agent roles
    agents_to_improve = snapshot.get("agents_to_improve", [])
    helper_agents = snapshot.get("helper_agents", [])
    all_agent_names = set(agents_to_improve + helper_agents)

    # Work with a copy of trajectories
    all_trajectories = snapshot["all_trajectories"].copy()

    # Pruning (same as standard reflection)
    if not snapshot.get("disable_trajectory_pruning", False):
        all_trajectories = _prune_trajectory_user_history(all_trajectories)
    
    if not snapshot.get("disable_system_prompt_pruning", False):
        for agent_name in helper_agents:
            if agent_name in all_trajectories and "system_prompt" in all_trajectories[agent_name]:
                del all_trajectories[agent_name]["system_prompt"]

    # --- Load Current Examples Context ---
    # The Exemplifier needs to know what examples already exist so it can delete bad ones
    # or avoid duplicates.
    current_examples_context = {}
    for agent_name in agents_to_improve:
        examples_path = os.path.join(settings.AGENT_BASE_DIR, f"agents/prompts/definitions/{agent_name.lower()}/reflection_examples.yaml")
        current_examples = []
        if os.path.exists(examples_path):
            try:
                with open(examples_path, "r") as f:
                    data = yaml.safe_load(f)
                    if data and "examples" in data:
                        # We only need names and descriptions for context to save tokens
                        current_examples = [
                            f"- Name: {ex.get('name')}\n  Description: {ex.get('description')}" 
                            for ex in data["examples"]
                        ]
            except Exception:
                pass
        current_examples_context[agent_name] = "\n".join(current_examples) if current_examples else "None"

    current_examples_str = ""
    for agent, examples in current_examples_context.items():
        current_examples_str += f"AGENT: {agent}\n{examples}\n\n"


    # --- Prompt Construction ---
    trajectories_section_str = "<TRAJECTORIES>\n"
    for agent_name in all_agent_names:
        trajectory = all_trajectories.get(agent_name, {})
        trajectories_section_str += f"  <AGENT_TRAJECTORY name='{agent_name}'>\n"
        trajectories_section_str += f"    {json.dumps(trajectory, indent=2)}\n"
        trajectories_section_str += f"  </AGENT_TRAJECTORY>\n"
    trajectories_section_str += "</TRAJECTORIES>"

    with open(os.path.join(settings.AGENT_BASE_DIR, "agents/prompts/definitions/exemplifier/initial_prompt.yaml"), "r") as f:
        prompt_data = yaml.safe_load(f)
        prompt_template = prompt_data.get("prompt", {}).get("opening", "") + "\n" + prompt_data.get("prompt", {}).get("closing", "")

    agents_to_improve_str = "\n".join(f"- {name}" for name in agents_to_improve)

    full_prompt = prompt_template.format(
        failed_step=snapshot["failed_step"],
        reward=snapshot["reward"],
        reward_threshold=snapshot.get("reward_threshold", -0.1),
        max_reflection_examples=snapshot.get("max_reflection_examples", 5),
        agents_to_improve=agents_to_improve_str,
        current_examples_list=current_examples_str,
        trajectories=trajectories_section_str
    )

    # --- Logging & Agent Setup ---
    logger = logging.getLogger(__name__)
    logger_name = "Exemplifier.Unified"
    log_file_path = os.path.join(output_dir, f"{logger_name}.log")
    add_dynamic_logger(logger_name, log_file_path)
    task_logger = logging.getLogger(logger_name)

    trajectories_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(trajectories_dir, exist_ok=True)
    exemplifier_log_path = os.path.join(trajectories_dir, "Exemplifier_Unified.json")

    exemplifier_executor = ExemplifierToolExecutor(logger=task_logger)
    
    dynamic_system_prompt = get_dynamic_exemplifier_system_prompt()

    exemplifier_agent = ReActAgent(
        system_prompt=dynamic_system_prompt,
        llm_provider=settings.EXEMPLIFIER_PROVIDER,
        llm_provider_parameters=EXEMPLIFIER_PROVIDER_PARAMETERS,
        logger=task_logger,
        log_name=logger_name,
        episode_step=current_attempt,
        trajectory_log_path=exemplifier_log_path,
    )

    # --- Run & Validate ---
    max_retries = 5
    final_json_output = None
    next_prompt = full_prompt

    for i in range(max_retries):
        raw_output = exemplifier_agent.run(next_prompt, exemplifier_executor)
        cleaned_output = _clean_heuristic(raw_output)

        try:
            parsed_json = json.loads(cleaned_output)
            
            # Validation
            for agent_name in agents_to_improve:
                if agent_name not in parsed_json:
                     # It's okay if not all agents get new examples, but the key should exist 
                     # if the prompt asked for it. However, strict enforcement helps.
                     # Let's be lenient: if key missing, assume no changes.
                     continue
                
                agent_data = parsed_json[agent_name]
                # Apply defaults for missing keys instead of failing validation
                # The LLM sometimes omits these keys when there are no changes
                agent_data.setdefault("delete_names", [])
                agent_data.setdefault("add_examples", [])
                
                # Deep validation of added examples
                for ex in agent_data["add_examples"]:
                    if "name" not in ex or "steps" not in ex:
                        raise ValueError(f"An added example for '{agent_name}' is missing 'name' or 'steps'.")
                    
                    for step in ex["steps"]:
                        if "type" not in step:
                            raise ValueError(f"A step in example '{ex.get('name')}' is missing 'type'.")
                        
                        step_type = step["type"]
                        if step_type == "tool_call":
                            if "name" not in step:
                                raise ValueError(f"A tool_call step in example '{ex.get('name')}' is missing 'name'.")
                        elif step_type in ["thought", "observation", "answer", "custom"]:
                            if "content" not in step:
                                raise ValueError(f"A '{step_type}' step in example '{ex.get('name')}' is missing 'content'.")
                        else:
                            raise ValueError(f"Invalid step type '{step_type}' in example '{ex.get('name')}'.")

            final_json_output = parsed_json
            task_logger.info(f"Successfully generated valid examples on attempt {i+1}.")
            break

        except (json.JSONDecodeError, ValueError) as e:
            task_logger.warning(f"Attempt {i+1} failed validation: {e}")
            next_prompt = (
                f"CRITICAL: Your previous response was NOT valid JSON. Error: {e}.\n\n"
                f"RULES YOU MUST FOLLOW:\n"
                f"1. NO MARKDOWN - Do not wrap JSON in ```json blocks\n"
                f"2. NO COMMENTS - JSON does not support comments (// or /* */)\n"
                f"3. Output ONLY the raw JSON object, nothing else\n\n"
                f"Ensure all examples have 'name', 'description', and valid 'steps'.\n"
                f"Previous Invalid Output:\n{cleaned_output}"
            )

    if not final_json_output:
        task_logger.error("Failed to generate valid examples after retries.")
        final_json_output = {}

    return final_json_output
