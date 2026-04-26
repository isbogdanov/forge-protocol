import logging
from typing import Dict, Any, List
import os
import json
import yaml
from concurrent.futures import ThreadPoolExecutor

from base_classes.react_agent_base import ReActAgent
from logs.config.log_config import get_current_run_dir, add_dynamic_logger
from agents.prompts.reflector import REFLECTOR_SYSTEM_PROMPT
from agents.reflector_tool_executor import ReflectorToolExecutor
import utils.settings as settings
from utils.settings import (
    # PROVIDER, # now available as settings.PROVIDER
    PROVIDER_PARAMETERS,
    # REFLECTOR_PROVIDER, # now available as settings.REFLECTOR_PROVIDER
    REFLECTOR_PROVIDER_PARAMETERS,
)
from coordinators.utils.reflection_utils import (
    run_unified_reflection_task,
    run_example_generation_task
)


class LearningCoordinator:
    def __init__(
        self,
        learnable_agents: List[str],
        helper_agents: List[str],
        agent_log_map: Dict[str, str],
        max_attempts: int = 10,
        reward_threshold: float = -0.5,
        session_dir: str = None,
        metrics_tracker=None,
        success_attempts_required: int = 1,
        max_reflection_rules: int = 100,
        learning_strategy: str = "rules",
        max_reflection_examples: int = 5,
    ):
        self.logger = logging.getLogger("LearningCoordinator")
        self.learnable_agents = learnable_agents
        self.helper_agents = helper_agents
        self.agent_log_map = agent_log_map
        self.max_attempts = max_attempts
        self.reward_threshold = reward_threshold
        self.session_dir = session_dir
        self.metrics_tracker = metrics_tracker
        self.success_attempts_required = success_attempts_required
        self.max_reflection_rules = max_reflection_rules
        self.learning_strategy = learning_strategy
        self.max_reflection_examples = max_reflection_examples

        self.current_attempt = 1
        self.successful_attempts_count = 0
        self.session_learning_history: List[Dict[str, Any]] = []
        self.session_reflection_history: List[str] = []
        self.logger.info(
            f"Learning Coordinator initialized for agents: {self.learnable_agents} (Helpers: {self.helper_agents})."
        )
        self.logger.info(
            f"Session will stop after {self.success_attempts_required} successful attempts or {self.max_attempts} total attempts."
        )
        self.logger.info(f"Learning Strategy: {self.learning_strategy.upper()}")

    def save_session_results(self):
        """Saves session metrics to results.json."""
        results = {
            "success": self.successful_attempts_count >= self.success_attempts_required,
            "successful_attempts": self.successful_attempts_count,
            "total_attempts": self.current_attempt,
            "max_attempts": self.max_attempts,
            "success_requirement": self.success_attempts_required,
            "learning_strategy": self.learning_strategy,
            "learning_history_length": len(self.session_learning_history)
        }
        
        if self.session_dir:
            results_path = os.path.join(self.session_dir, "results.json")
            try:
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)
                self.logger.info(f"Session results saved to {results_path}")
            except Exception as e:
                self.logger.error(f"Failed to save session results: {e}")

    def should_continue_attempts(self) -> bool:
        """Determines if the learning session should continue."""
        return (
            self.current_attempt <= self.max_attempts
            and self.successful_attempts_count < self.success_attempts_required
        )

    def get_current_attempt(self) -> int:
        """Returns the current attempt number."""
        return self.current_attempt

    def report_attempt_result(self, was_successful: bool):
        """Logs the result of an attempt and updates success counter."""
        if was_successful:
            self.successful_attempts_count += 1
            self.logger.info(
                f"--- Attempt #{self.current_attempt} was SUCCESSFUL --- ({self.successful_attempts_count}/{self.success_attempts_required} successes)"
            )
        else:
            self.logger.info(f"--- Attempt #{self.current_attempt} FAILED ---")

    def advance_attempt(self):
        """Increments the attempt counter."""
        self.current_attempt += 1

    def learn_from_failure(self, snapshot: Dict[str, Any]):
        """
        Orchestrates the learning process after a failure using a pre-compiled snapshot.
        This version supports both Rule-Based and Example-Based learning strategies.
        """
        failed_step = snapshot["failed_step"]
        reward = snapshot["reward"]
        self.logger.info(
            f"Attempt #{self.current_attempt} failed at step {failed_step} with reward {reward}. "
            f"Initiating unified learning process (Strategy: {self.learning_strategy})."
        )

        if self.learning_strategy == "rules":
            # Rule-Based Learning (Standard)
            unified_results = run_unified_reflection_task(
                snapshot, self.session_dir, self.current_attempt
            )
            self._process_rule_based_results(unified_results, failed_step, reward)
        
        elif self.learning_strategy == "examples":
            # Example-Based Learning (Exemplifier)
            unified_results = run_example_generation_task(
                snapshot, self.session_dir, self.current_attempt
            )
            self._process_example_based_results(unified_results, failed_step, reward)

        elif self.learning_strategy == "mixed":
            # Hybrid Learning: Run both sequentially
            self.logger.info("Executing Hybrid Learning (Rules + Examples)...")
            
            # 1. Run Reflector (Rules)
            self.logger.info("Step 1/2: Running Reflector for Rules...")
            rule_results = run_unified_reflection_task(
                snapshot, self.session_dir, self.current_attempt
            )
            self._process_rule_based_results(rule_results, failed_step, reward)
            
            # 2. Run Exemplifier (Examples)
            self.logger.info("Step 2/2: Running Exemplifier for Examples...")
            example_results = run_example_generation_task(
                snapshot, self.session_dir, self.current_attempt
            )
            self._process_example_based_results(example_results, failed_step, reward)
        
        else:
            self.logger.error(f"Unknown learning strategy: {self.learning_strategy}")

        self.logger.info("Advancing to the next attempt.")

    def _process_rule_based_results(self, unified_results, failed_step, reward):
        for agent_name, operations in unified_results.items():
            if not operations or not isinstance(operations, dict):
                self.logger.warning(
                    f"No valid operations found for agent {agent_name} in unified result."
                )
                continue

            heuristic_json_str = json.dumps(operations)
            self._persist_heuristic(agent_name, heuristic_json_str)
            self._append_learning_history(agent_name, operations, failed_step, reward)

    def _process_example_based_results(self, unified_results, failed_step, reward):
        for agent_name, operations in unified_results.items():
            if not operations or not isinstance(operations, dict):
                 # Logging already handled in run_example_generation_task
                 continue
            
            self._persist_reflection_examples(agent_name, operations)
            
            # History logging for examples
            # We adapt the structure slightly since examples are structured differently
            history_entry = {
                 "attempt_number": self.current_attempt,
                 "failed_at_step": failed_step,
                 "failure_reward": reward,
                 "agent_name": agent_name,
                 "rationale": operations.get("rationale", "No rationale provided."),
                 "deleted_examples": operations.get("delete_names", []),
                 "added_examples_count": len(operations.get("add_examples", []))
            }
            self.session_learning_history.append(history_entry)

    def _append_learning_history(self, agent_name, operations, failed_step, reward):
        try:
            self.session_learning_history.append(
                {
                    "attempt_number": self.current_attempt,
                    "failed_at_step": failed_step,
                    "failure_reward": reward,
                    "agent_name": agent_name,
                    "rationale": operations.get(
                        "rationale", "No rationale provided."
                    ),
                    "deleted_rules": operations.get("delete", []),
                    "added_rules": operations.get("add", []),
                }
            )
        except (json.JSONDecodeError, AttributeError):
             pass

    def _persist_heuristic(self, agent_name: str, heuristic_json_str: str):
        """
        Parses a JSON string with 'delete' and 'add' commands to intelligently
        update the agent's knowledge files.
        """
        try:
            operations = json.loads(heuristic_json_str)
            deletions = operations.get("delete", [])
            additions = operations.get("add", [])
        except json.JSONDecodeError:
            self.logger.error(
                f"Invalid JSON from Reflector for {agent_name}. Raw output: {heuristic_json_str}"
            )
            # Fallback for non-JSON output: treat it as a simple addition
            deletions = []
            additions = [heuristic_json_str]

        # Define paths for both editable knowledge files
        reflection_path = os.path.join(settings.AGENT_BASE_DIR, 
            f"agents/prompts/definitions/{agent_name.lower()}/reflection_knowledge.yaml"
        )
        tactical_path = os.path.join(settings.AGENT_BASE_DIR,
            f"agents/prompts/definitions/{agent_name.lower()}/tactical_knowledge.yaml"
        )

        try:
            # Load current reflection knowledge
            with open(reflection_path, "r") as f:
                reflection_data = yaml.safe_load(f) or {"reflection_knowledge": []}
            reflection_rules = reflection_data.get("reflection_knowledge") or []

            # Load current tactical knowledge
            if os.path.exists(tactical_path):
                with open(tactical_path, "r") as f:
                    tactical_data = yaml.safe_load(f) or {"reflection_knowledge": []}
                tactical_rules = tactical_data.get("reflection_knowledge") or []
            else:
                tactical_rules = []

            # Combine to form the list that the Reflector saw (and numbered)
            editable_knowledge = reflection_rules + tactical_rules

            # Process deletions
            # Handle both integer indices (1-based) and string identifiers/content
            delete_indices = set()
            for d in deletions:
                try:
                    # Case 1: Integer index (or string digit)
                    if isinstance(d, int):
                        delete_indices.add(d - 1)
                    elif isinstance(d, str) and d.isdigit():
                        delete_indices.add(int(d) - 1)
                    elif isinstance(d, str):
                        # Case 2: String identifier (e.g., "heuristic-P-2")
                        # Find the rule that starts with this ID or matches exactly
                        d_str = d.strip()
                        found = False
                        for idx, rule in enumerate(editable_knowledge):
                            if rule.strip().startswith(d_str) or rule.strip() == d_str:
                                delete_indices.add(idx)
                                found = True
                                break
                        if not found:
                            self.logger.warning(f"Could not find rule to delete matching: '{d_str}'")
                except Exception as e:
                    self.logger.warning(f"Error processing deletion '{d}': {e}")

            updated_knowledge = [
                rule
                for i, rule in enumerate(editable_knowledge)
                if i not in delete_indices
            ]

            # Process additions
            updated_knowledge.extend(additions)

            # Enforce the rule limit, keeping the newest rules
            if (
                self.max_reflection_rules != -1
                and len(updated_knowledge) > self.max_reflection_rules
            ):
                updated_knowledge = updated_knowledge[-self.max_reflection_rules :]
                self.logger.info(
                    f"Knowledge base for {agent_name} truncated to the newest {self.max_reflection_rules} rules."
                )

            # In the offline learning loop, we consolidate all learned knowledge
            # into the main reflection file and clear the tactical one.
            final_knowledge_data = {"reflection_knowledge": updated_knowledge}
            with open(reflection_path, "w") as f:
                yaml.dump(final_knowledge_data, f, indent=2, default_flow_style=False)

            # Clear the tactical knowledge file
            if os.path.exists(tactical_path):
                with open(tactical_path, "w") as f:
                    yaml.dump({"reflection_knowledge": []}, f)

            self.logger.info(
                f"Successfully updated knowledge for {agent_name}: "
                f"{len(deletions)} rules deleted, {len(additions)} rules added. "
                f"Knowledge consolidated into {reflection_path}."
            )

        except (FileNotFoundError, IOError, yaml.YAMLError) as e:
            self.logger.error(
                f"Could not persist heuristic for '{agent_name}'. Error: {e}"
            )

    def _persist_reflection_examples(self, agent_name: str, operations: Dict[str, Any]):
        """
        Updates the agent's reflection_examples.yaml based on the Exemplifier's output.
        """
        delete_names = set(operations.get("delete_names", []))
        add_examples = operations.get("add_examples", [])
        
        examples_path = os.path.join(settings.AGENT_BASE_DIR, f"agents/prompts/definitions/{agent_name.lower()}/reflection_examples.yaml")
        
        try:
            current_data = {}
            if os.path.exists(examples_path):
                with open(examples_path, "r") as f:
                    current_data = yaml.safe_load(f) or {}
            
            current_examples = current_data.get("examples") or []
            
            # Process Deletions (by name)
            updated_examples = [
                ex for ex in current_examples 
                if ex.get("name") not in delete_names
            ]
            
            # Process Additions
            updated_examples.extend(add_examples)
            
            # Enforce Limit (Keep newest)
            if self.max_reflection_examples != -1 and len(updated_examples) > self.max_reflection_examples:
                updated_examples = updated_examples[-self.max_reflection_examples:]
                self.logger.info(
                    f"Example list for {agent_name} truncated to the newest {self.max_reflection_examples} examples."
                )
            
            # Save
            final_data = {"examples": updated_examples}
            with open(examples_path, "w") as f:
                yaml.dump(final_data, f, indent=2, default_flow_style=False, sort_keys=False)
                
            self.logger.info(
                f"Successfully updated reflection examples for {agent_name}: "
                f"{len(delete_names)} deleted, {len(add_examples)} added. "
                f"Persisted to {examples_path}."
            )
            
        except Exception as e:
            self.logger.error(f"Failed to persist reflection examples for {agent_name}: {e}")

    def _clean_heuristic(self, raw_heuristic: str) -> str:
        """Cleans the raw output from the Reflector agent."""
        if not isinstance(raw_heuristic, str):
            return "Error: Invalid heuristic format."

        if "Answer:" in raw_heuristic:
            answer_pos = raw_heuristic.find("Answer:")
            raw_heuristic = raw_heuristic[answer_pos + len("Answer:") :].strip()

        raw_heuristic = raw_heuristic.strip()
        if raw_heuristic.startswith("Thought:"):
            raw_heuristic = raw_heuristic[len("Thought:") :].strip()

        raw_heuristic = raw_heuristic.replace("PAUSE", "").strip()
        return raw_heuristic
