from logs.config.log_config import get_context_logger
import queue
import threading
import os
import yaml
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json

from coordinators.utils.reflection_utils import run_unified_reflection_task
from utils.settings import ONLINE_LEARNING_QUEUE_MAX_SIZE


class OnlineLearningManager:
    def __init__(
        self,
        session_dir: str,
        agents_to_improve: List[str],
        helper_agents: List[str],
        agent_log_map: Dict[str, str],
        queue_max_size: int = 5,
        max_reflection_rules: int = 100,
    ):
        self.logger = logging.getLogger("OnlineLearningManager")
        self.session_dir = session_dir
        self.agents_to_improve = agents_to_improve
        self.helper_agents = helper_agents
        self.agent_log_map = agent_log_map
        self.queue_max_size = queue_max_size
        self.max_reflection_rules = max_reflection_rules

        self.snapshot_queue = queue.Queue(maxsize=self.queue_max_size)
        self.file_write_lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._learning_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        self.logger.info(
            f"Online Learning Manager initialized with queue size {self.queue_max_size}."
        )

    def submit_failure_snapshot(self, snapshot: Dict[str, Any]):
        try:
            self.snapshot_queue.put_nowait(snapshot)
            self.logger.info(
                f"Received failure snapshot for step {snapshot.get('failed_step')}. Submitting to learning queue."
            )
        except queue.Full:
            self.logger.warning(
                f"Online learning queue is full (max size: {self.queue_max_size}). "
                f"Dropping snapshot for failed step {snapshot.get('failed_step')}."
            )

    def _learning_worker(self):
        while True:
            snapshot = self.snapshot_queue.get()
            if snapshot is None:  # Sentinel value for shutting down
                self.logger.info("Shutdown signal received. Worker thread terminating.")
                break

            # Create a unique, timestamped directory for this learning event
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            snapshot_run_dir = os.path.join(
                self.session_dir, "online_learning", f"run_{timestamp}"
            )
            os.makedirs(snapshot_run_dir, exist_ok=True)

            # Save the snapshot data for debugging and traceability
            with open(os.path.join(snapshot_run_dir, "snapshot_data.json"), "w") as f:
                # We need a custom serializer for dataclasses like EpisodeMemory
                f.write(
                    json.dumps(
                        snapshot,
                        default=lambda o: (
                            o.__dict__ if hasattr(o, "__dict__") else str(o)
                        ),
                        indent=2,
                    )
                )

            self.logger.info(
                f"Processing snapshot for failed step {snapshot.get('failed_step')}. "
                f"Logs will be in: {snapshot_run_dir}"
            )

            # A single call to the unified reflection task
            unified_results = run_unified_reflection_task(
                snapshot, snapshot_run_dir, snapshot["failed_step"]
            )

            # Process the unified results dictionary
            for agent_name, operations in unified_results.items():
                if operations and isinstance(operations, dict):
                    self._persist_tactical_heuristic(agent_name, json.dumps(operations))

            self.snapshot_queue.task_done()

    def _persist_tactical_heuristic(self, agent_name: str, heuristic_json_str: str):
        """
        Parses a JSON string with 'delete' and 'add' commands to intelligently
        update the agent's tactical knowledge file using a thread-safe method.
        """
        try:
            operations = json.loads(heuristic_json_str)
            deletions = operations.get("delete", [])
            additions = operations.get("add", [])
        except json.JSONDecodeError:
            self.logger.error(
                f"Invalid JSON from Reflector for {agent_name}. Raw output: {heuristic_json_str}"
            )
            deletions = []
            additions = [heuristic_json_str]

        # Define paths for knowledge files
        reflection_path = (
            os.path.join(settings.AGENT_BASE_DIR, f"agents/prompts/definitions/{agent_name.lower()}/reflection_knowledge.yaml")
        )
        tactical_path = (
            os.path.join(settings.AGENT_BASE_DIR, f"agents/prompts/definitions/{agent_name.lower()}/tactical_knowledge.yaml")
        )
        temp_path = f"{tactical_path}.tmp"

        with self.file_write_lock:
            try:
                # Load reflection knowledge (read-only, for context)
                if os.path.exists(reflection_path):
                    with open(reflection_path, "r") as f:
                        reflection_data = yaml.safe_load(f) or {
                            "reflection_knowledge": []
                        }
                    reflection_rules = reflection_data.get("reflection_knowledge") or []
                else:
                    reflection_rules = []

                # Load current tactical knowledge (this is what we'll modify)
                if os.path.exists(tactical_path):
                    with open(tactical_path, "r") as f:
                        tactical_data = yaml.safe_load(f) or {
                            "reflection_knowledge": []
                        }
                    tactical_rules = tactical_data.get("reflection_knowledge") or []
                else:
                    tactical_rules = []

                # Process deletions
                delete_indices = {idx - 1 for idx in deletions}
                # The full list of rules, as the Reflector saw it
                combined_knowledge = reflection_rules + tactical_rules

                # Filter the combined list to get the rules that should remain
                updated_knowledge = [
                    rule
                    for i, rule in enumerate(combined_knowledge)
                    if i not in delete_indices
                ]

                # Add the new heuristics
                updated_knowledge.extend(additions)

                # Now, filter this updated list to get only the rules that should be
                # in the tactical file. These are any rules that are NOT in the
                # original, stable reflection_rules list.
                reflection_rules_set = set(reflection_rules)
                updated_tactical_knowledge = [
                    rule
                    for rule in updated_knowledge
                    if rule not in reflection_rules_set
                ]

                # Enforce the rule limit on the tactical knowledge
                if (
                    self.max_reflection_rules != -1
                    and len(updated_tactical_knowledge) > self.max_reflection_rules
                ):
                    updated_tactical_knowledge = updated_tactical_knowledge[
                        -self.max_reflection_rules :
                    ]
                    self.logger.info(
                        f"Tactical knowledge for {agent_name} truncated to the newest {self.max_reflection_rules} rules."
                    )

                # Atomically write the updated rules back to the TACTICAL file
                final_knowledge_data = {
                    "reflection_knowledge": updated_tactical_knowledge
                }
                with open(temp_path, "w") as f:
                    yaml.dump(
                        final_knowledge_data, f, indent=2, default_flow_style=False
                    )
                os.rename(temp_path, tactical_path)

                self.logger.info(
                    f"Successfully updated tactical knowledge for {agent_name}: "
                    f"{len(deletions)} deletions attempted, {len(additions)} rules added to {tactical_path}"
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to persist tactical heuristic for {agent_name}: {e}"
                )
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    def shutdown(self):
        self.logger.info("Shutting down Online Learning Manager.")
        self.snapshot_queue.put(None)  # Send sentinel
        self.worker_thread.join()
        self.logger.info("Worker thread has been joined. Shutdown complete.")
