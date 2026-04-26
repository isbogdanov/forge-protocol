import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LearningMetricsTracker:
    """Tracks learning metrics across attempts for continual learning sessions."""

    def __init__(self, session_dir: str):
        """
        Initialize metrics tracker for a learning session.

        Args:
            session_dir: Directory path for the current learning session
        """
        self.session_dir = session_dir
        self.metrics_file = os.path.join(session_dir, "learning_metrics.json")
        self.metrics_data = self._load_or_initialize_metrics()

    def _load_or_initialize_metrics(self) -> Dict[str, Any]:
        """Load existing metrics or initialize new metrics structure."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"Could not load existing metrics: {e}. Initializing new metrics."
                )

        # Initialize new metrics structure
        return {
            "session_info": {
                "session_start_time": datetime.now().isoformat(),
                "session_dir": self.session_dir,
                "total_attempts": 0,
                "successful_attempts": 0,
                "failed_attempts": 0,
            },
            "attempts": {},
        }

    def start_attempt(self, attempt_number: int, max_steps: int) -> None:
        """Record the start of a new attempt."""
        self.metrics_data["attempts"][str(attempt_number)] = {
            "attempt_number": attempt_number,
            "attempt_start_time": datetime.now().isoformat(),
            "max_steps": max_steps,
            "steps_completed": 0,
            "total_reward": 0.0,
            "step_rewards": [],
            "actions_taken": [],
            "reflection_triggered": False,
            "reflection_step": None,
            "reflection_reason": None,
            "attempt_status": "in_progress",
            "attempt_end_time": None,
            "failure_reason": None,
        }

        # Update session totals
        self.metrics_data["session_info"]["total_attempts"] = len(
            self.metrics_data["attempts"]
        )
        self._save_metrics()

        logger.info(f"Started tracking metrics for attempt {attempt_number}")

    def log_step(
        self, attempt_number: int, step: int, action: str, reward: float
    ) -> None:
        """Log metrics for a completed step."""
        attempt_key = str(attempt_number)
        if attempt_key not in self.metrics_data["attempts"]:
            logger.error(
                f"Attempt {attempt_number} not found in metrics. Call start_attempt() first."
            )
            return

        attempt_data = self.metrics_data["attempts"][attempt_key]
        attempt_data["steps_completed"] = step
        attempt_data["total_reward"] += reward
        attempt_data["step_rewards"].append(
            {
                "step": step,
                "reward": reward,
                "cumulative_reward": attempt_data["total_reward"],
            }
        )
        attempt_data["actions_taken"].append(
            {
                "step": step,
                "action": str(action),
                "timestamp": datetime.now().isoformat(),
            }
        )

        self._save_metrics()

        logger.debug(
            f"Logged step {step} for attempt {attempt_number}: action={action}, reward={reward}"
        )

    def log_reflection(self, attempt_number: int, step: int, reason: str) -> None:
        """Log that reflection was triggered."""
        attempt_key = str(attempt_number)
        if attempt_key not in self.metrics_data["attempts"]:
            logger.error(f"Attempt {attempt_number} not found in metrics.")
            return

        attempt_data = self.metrics_data["attempts"][attempt_key]
        attempt_data["reflection_triggered"] = True
        attempt_data["reflection_step"] = step
        attempt_data["reflection_reason"] = reason

        self._save_metrics()

        logger.info(
            f"Logged reflection for attempt {attempt_number} at step {step}: {reason}"
        )

    def complete_attempt(
        self, attempt_number: int, success: bool, failure_reason: Optional[str] = None
    ) -> None:
        """Mark an attempt as completed."""
        attempt_key = str(attempt_number)
        if attempt_key not in self.metrics_data["attempts"]:
            logger.error(f"Attempt {attempt_number} not found in metrics.")
            return

        attempt_data = self.metrics_data["attempts"][attempt_key]
        attempt_data["attempt_status"] = "success" if success else "failed"
        attempt_data["attempt_end_time"] = datetime.now().isoformat()

        if not success and failure_reason:
            attempt_data["failure_reason"] = failure_reason

        # Update session totals
        session_info = self.metrics_data["session_info"]
        if success:
            session_info["successful_attempts"] = (
                session_info.get("successful_attempts", 0) + 1
            )
        else:
            session_info["failed_attempts"] = session_info.get("failed_attempts", 0) + 1

        self._save_metrics()

        status = "successfully" if success else f"with failure: {failure_reason}"
        logger.info(f"Completed attempt {attempt_number} {status}")

    def get_attempt_summary(self, attempt_number: int) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a specific attempt."""
        attempt_key = str(attempt_number)
        if attempt_key not in self.metrics_data["attempts"]:
            return None

        attempt_data = self.metrics_data["attempts"][attempt_key]

        return {
            "attempt_number": attempt_data["attempt_number"],
            "steps_completed": attempt_data["steps_completed"],
            "max_steps": attempt_data["max_steps"],
            "total_reward": attempt_data["total_reward"],
            "average_reward": attempt_data["total_reward"]
            / max(attempt_data["steps_completed"], 1),
            "reflection_triggered": attempt_data["reflection_triggered"],
            "reflection_step": attempt_data.get("reflection_step"),
            "status": attempt_data["attempt_status"],
        }

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the entire session."""
        attempts = list(self.metrics_data["attempts"].values())

        if not attempts:
            return {"message": "No attempts recorded yet"}

        # Calculate aggregate statistics
        total_steps = sum(a["steps_completed"] for a in attempts)
        total_rewards = [a["total_reward"] for a in attempts]
        successful_attempts = [a for a in attempts if a["attempt_status"] == "success"]

        return {
            "session_info": self.metrics_data["session_info"],
            "total_attempts": len(attempts),
            "successful_attempts": len(successful_attempts),
            "success_rate": len(successful_attempts) / len(attempts) if attempts else 0,
            "total_steps_across_all_attempts": total_steps,
            "average_steps_per_attempt": total_steps / len(attempts) if attempts else 0,
            "total_rewards": total_rewards,
            "best_reward": max(total_rewards) if total_rewards else 0,
            "worst_reward": min(total_rewards) if total_rewards else 0,
            "average_reward": (
                sum(total_rewards) / len(total_rewards) if total_rewards else 0
            ),
            "reflections_triggered": sum(
                1 for a in attempts if a["reflection_triggered"]
            ),
        }

    def export_for_plotting(self) -> Dict[str, Any]:
        """Export data in a format suitable for plotting."""
        attempts = []
        for attempt_data in self.metrics_data["attempts"].values():
            attempts.append(
                {
                    "attempt": attempt_data["attempt_number"],
                    "steps": attempt_data["steps_completed"],
                    "total_reward": attempt_data["total_reward"],
                    "average_reward": attempt_data["total_reward"]
                    / max(attempt_data["steps_completed"], 1),
                    "reflection_triggered": attempt_data["reflection_triggered"],
                    "reflection_step": attempt_data.get("reflection_step"),
                    "status": attempt_data["attempt_status"],
                    "step_rewards": [
                        sr["reward"] for sr in attempt_data["step_rewards"]
                    ],
                    "cumulative_rewards": [
                        sr["cumulative_reward"] for sr in attempt_data["step_rewards"]
                    ],
                }
            )

        return {"session_summary": self.get_session_summary(), "attempts": attempts}

    def _save_metrics(self) -> None:
        """Save metrics to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
            with open(self.metrics_file, "w") as f:
                json.dump(self.metrics_data, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save metrics: {e}")
