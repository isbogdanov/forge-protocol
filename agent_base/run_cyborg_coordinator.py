import logging
import inspect
import argparse
import os
from datetime import datetime
from typing import List

# from llm_cage_agent import LLMCageAgent
from CybORG import CybORG
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Agents.Wrappers import BlueTableWrapper
from CybORG.Agents import RedMeanderAgent
from CybORG.Agents import B_lineAgent

from CybORG.Agents import SleepAgent

from CybORG.Agents.SimpleAgents.BlueReactAgent import (
    BlueReactRemoveAgent,
    BlueReactRestoreAgent,
)
from CybORG.Agents.SimpleAgents.KeyboardAgent import KeyboardAgent

# Import action processing functions
from utils.helpers.action_processing import format_blue_action_space, get_blue_actions
from logs.config.log_config import setup_logging
import utils.settings as settings

# Import our custom ReactAgent
# from react_agent import ReactAgent

# from my_agent import MyAgent

# from blue_random_agent import BlueRandomAgent

from coordinators.cyborg_agent_coordinator import CybORGAgentCoordinator
from coordinators.learning_coordinator import LearningCoordinator
from coordinators.online_learning_manager import OnlineLearningManager
from utils.learning_metrics import LearningMetricsTracker


def main():
    parser = argparse.ArgumentParser(
        description="Run CybORG agent with optional continual learning."
    )
    parser.add_argument(
        "--online-learning",
        action="store_true",
        help="Enable online learning mode, which runs reflection in the background.",
    )
    parser.add_argument(
        "--online-learning-queue-size",
        type=int,
        default=5,
        help="Max number of failure events to queue for online learning.",
    )
    parser.add_argument(
        "--continual-learning",
        action="store_true",
        help="Enable continual learning mode with reflection.",
    )
    parser.add_argument(
        "--learning-strategy",
        type=str,
        default="rules",
        choices=["rules", "examples", "mixed"],
        help="Strategy for learning from failure: 'rules' (text-based heuristics) or 'examples' (few-shot gold standard examples).",
    )
    parser.add_argument(
        "--agents-to-improve",
        nargs="+",
        default=["action_chooser"],
        help="List of agent names to be improved during the learning session.",
    )
    parser.add_argument(
        "--helper-agents",
        nargs="*",
        default=[],
        help="List of agent names whose trajectories to include as context for reflection.",
    )
    parser.add_argument(
        "--steps", type=int, default=30, help="Number of steps per episode."
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=-0.1,
        help="Reward threshold to trigger reflection in learning mode.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Max number of attempts in a continual learning session.",
    )
    parser.add_argument(
        "--success-attempts",
        type=int,
        default=1,
        help="Number of successful attempts required to end the learning session.",
    )
    parser.add_argument(
        "--learning-on-step",
        type=int,
        default=1,
        help="The first step number where continual learning can be triggered.",
    )
    parser.add_argument(
        "--learning-off-step",
        type=int,
        default=None,
        help="The last step number where continual learning can be triggered. Defaults to total steps.",
    )
    parser.add_argument(
        "--max-reflection-rules",
        type=int,
        default=100,
        help="The maximum number of editable (reflection + tactical) rules an agent can have. -1 for unlimited.",
    )
    parser.add_argument(
        "--max-reflection-examples",
        type=int,
        default=10,
        help="The maximum number of learned examples an agent can have (for 'examples' strategy). -1 for unlimited.",
    )
    parser.add_argument(
        "--disable-trajectory-pruning",
        action="store_true",
        help="Disable pruning of user history from trajectories for reflection.",
    )
    parser.add_argument(
        "--disable-system-prompt-pruning",
        action="store_true",
        help="Disable pruning of system prompts from context trajectories for reflection.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Override the LLM provider (e.g., 'google', 'openai', 'openrouter').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the LLM model name.",
    )
    
    # Granular overrides for specific agents
    parser.add_argument(
        "--reflector-provider", type=str, default=None, help="Override Reflector LLM provider."
    )
    parser.add_argument(
        "--reflector-model", type=str, default=None, help="Override Reflector LLM model."
    )
    parser.add_argument(
        "--exemplifier-provider", type=str, default=None, help="Override Exemplifier LLM provider."
    )
    parser.add_argument(
        "--exemplifier-model", type=str, default=None, help="Override Exemplifier LLM model."
    )
    args = parser.parse_args()

    # Apply settings override if arguments are provided
    if args.provider or args.model:
        # Use existing defaults if only one argument is provided
        current_provider, current_model = settings.PROVIDER
        new_provider = args.provider if args.provider else current_provider
        new_model = args.model if args.model else current_model
        
        settings.update_provider_settings(new_provider, new_model)
        
        # Log the update using basic print since logger might not be setup yet or we want immediate feedback
        print(f"CLI Override: Settings updated to Provider='{new_provider}', Model='{new_model}'")
        
    # Apply Reflector overrides
    if args.reflector_provider or args.reflector_model:
        current_reflector = settings.REFLECTOR_PROVIDER
        new_ref_provider = args.reflector_provider if args.reflector_provider else current_reflector[0]
        new_ref_model = args.reflector_model if args.reflector_model else current_reflector[1]
        settings.update_reflector_settings(new_ref_provider, new_ref_model)
        print(f"CLI Override: Reflector updated to Provider='{new_ref_provider}', Model='{new_ref_model}'")

    # Apply Exemplifier overrides
    if args.exemplifier_provider or args.exemplifier_model:
        current_exemplifier = settings.EXEMPLIFIER_PROVIDER
        new_ex_provider = args.exemplifier_provider if args.exemplifier_provider else current_exemplifier[0]
        new_ex_model = args.exemplifier_model if args.exemplifier_model else current_exemplifier[1]
        settings.update_exemplifier_settings(new_ex_provider, new_ex_model)
        print(f"CLI Override: Exemplifier updated to Provider='{new_ex_provider}', Model='{new_ex_model}'")


    # If learning-off-step is not specified, it defaults to the total number of steps
    if args.learning_off_step is None:
        args.learning_off_step = args.steps

    # Define the mapping of agent names to their trajectory log files
    agent_log_map = {
        "action_chooser": "Planner.Executor.ActionChooser.json",
        "analyst": "Planner.Executor.Analyst.json",
        "planner": "Planner.json",
    }

    # Configure logging and determine session directory
    session_dir = None
    if args.continual_learning:
        learning_dir = os.path.join(settings.AGENT_BASE_DIR, "logs", "runs", "learning")
        os.makedirs(learning_dir, exist_ok=True)
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(
            learning_dir, f"learning_session_{session_timestamp}"
        )
        os.makedirs(session_dir, exist_ok=True)
        # In learning mode, initial logging is set up inside the loop
    else:
        # For evaluation, create a single run directory inside 'evaluating'
        eval_dir = os.path.join(settings.AGENT_BASE_DIR, "logs", "runs", "evaluating")
        os.makedirs(eval_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(eval_dir, f"run_{timestamp}")
        session_dir = run_dir  # For eval, session and attempt dirs are the same
        setup_logging(attempt_dir=run_dir, session_dir=run_dir)

    logger = logging.getLogger(__name__)

    # Initialize Online Learning Manager if enabled
    online_learning_manager = None
    if args.online_learning:
        online_learning_manager = OnlineLearningManager(
            session_dir=session_dir,
            agents_to_improve=args.agents_to_improve,
            helper_agents=args.helper_agents,
            agent_log_map=agent_log_map,
            queue_max_size=args.online_learning_queue_size,
            max_reflection_rules=args.max_reflection_rules,
        )

    try:
        # Get path to scenario file
        cyborg_module_path = os.path.dirname(inspect.getfile(CybORG))
        scenario_path = os.path.join(cyborg_module_path, "Shared", "Scenarios", "Scenario2.yaml")

        # Initialize the environment
        logger.info(f"Initializing CAGE environment with scenario: {scenario_path}")
        cyborg = CybORG(
            scenario_path,
            "sim",
            agents={"Red": B_lineAgent},
            # agents={"Red": SleepAgent},
            # agents={"Red": RedMeanderAgent},
        )

        # Create our custom ReactAgent
        # agent = ReactAgent()
        # agent = SleepAgent()
        # agent = CybORGAgentCoordinator()
        # agent = BlueReactRemoveAgent()
        # agent = KeyboardAgent()
        # agent = MyAgent()
        # agent = BlueRandomAgent()

        if args.continual_learning:
            run_continual_learning_session(
                cyborg,
                steps=args.steps,
                reward_threshold=args.reward_threshold,
                max_attempts=args.max_attempts,
                agents_to_improve=args.agents_to_improve,
                helper_agents=args.helper_agents,
                success_attempts_required=args.success_attempts,
                online_learning_manager=online_learning_manager,
                agent_log_map=agent_log_map,
                session_dir=session_dir,
                learning_on_step=args.learning_on_step,
                learning_off_step=args.learning_off_step,
                max_reflection_rules=args.max_reflection_rules,
                disable_trajectory_pruning=args.disable_trajectory_pruning,
                disable_system_prompt_pruning=args.disable_system_prompt_pruning,
                learning_strategy=args.learning_strategy,
                max_reflection_examples=args.max_reflection_examples,
            )
        else:
            # Agent is created here, after initial logging setup.
            agent = CybORGAgentCoordinator(log_summary=True)
            run_evaluation_session(
                cyborg,
                agent,
                steps=args.steps,
                online_learning_manager=online_learning_manager,
                online_learning_args={
                    "agents_to_improve": args.agents_to_improve,
                    "helper_agents": args.helper_agents,
                    "agent_log_map": agent_log_map,
                    "max_reflection_rules": args.max_reflection_rules,
                },
                reward_threshold=args.reward_threshold,
                learning_on_step=args.learning_on_step,
                learning_off_step=args.learning_off_step,
                disable_trajectory_pruning=args.disable_trajectory_pruning,
                disable_system_prompt_pruning=args.disable_system_prompt_pruning,
            )

    finally:
        if online_learning_manager:
            online_learning_manager.shutdown()


def run_evaluation_session(
    cyborg: CybORG,
    agent: CybORGAgentCoordinator,
    n_episodes=1,
    steps=30,
    online_learning_manager=None,
    online_learning_args=None,
    reward_threshold=-0.1,
    learning_on_step: int = 1,
    learning_off_step: int = 30,
    disable_trajectory_pruning: bool = False,
    disable_system_prompt_pruning: bool = False,
):
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Standard Evaluation Session ---")
    episode_rewards = []
    agent_name = agent.__class__.__name__
    logger.info(f"Running {n_episodes} episodes with {agent_name}...")

    for episode in range(n_episodes):
        logger.info(f"Starting Episode {episode + 1}/{n_episodes}")
        total_reward, _, _, _ = run_single_episode(
            cyborg,
            agent,
            episode_num=episode + 1,
            steps=steps,
            is_learning_mode=True if online_learning_manager else False,
            online_learning_manager=online_learning_manager,
            online_learning_args=online_learning_args,
            reward_threshold=reward_threshold,
            learning_on_step=learning_on_step,
            learning_off_step=learning_off_step,
            disable_trajectory_pruning=disable_trajectory_pruning,
            disable_system_prompt_pruning=disable_system_prompt_pruning,
        )
        episode_rewards.append(total_reward)

    # Print summary of all episode rewards
    logger.info("===== EPISODE REWARDS SUMMARY =====")
    logger.info(f"Average Reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    logger.info(f"Total Cumulative Reward: {sum(episode_rewards):.2f}")
    logger.info(
        f"Best Episode: #{episode_rewards.index(max(episode_rewards))+1} with reward {max(episode_rewards):.2f}"
    )
    logger.info(
        f"Worst Episode: #{episode_rewards.index(min(episode_rewards))+1} with reward {min(episode_rewards):.2f}"
    )
    logger.info("Evaluation complete.")


def run_continual_learning_session(
    cyborg: CybORG,
    steps: int,
    reward_threshold: float,
    max_attempts: int,
    agents_to_improve: List[str],
    helper_agents: List[str],
    success_attempts_required: int,
    online_learning_manager=None,
    agent_log_map=None,
    session_dir=None,
    learning_on_step: int = 1,
    learning_off_step: int = 30,
    max_reflection_rules: int = 100,
    disable_trajectory_pruning: bool = False,
    disable_system_prompt_pruning: bool = False,
    learning_strategy: str = "rules",
    max_reflection_examples: int = 5,
):
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Continual Learning Session ---")

    # Initialize metrics tracker for this learning session
    metrics_tracker = LearningMetricsTracker(session_dir)
    logger.info(f"Learning metrics will be tracked in: {metrics_tracker.metrics_file}")

    # In learning mode, agent is created per attempt, without summary logging
    learning_coordinator = LearningCoordinator(
        learnable_agents=agents_to_improve,
        helper_agents=helper_agents,
        agent_log_map=agent_log_map,
        max_attempts=max_attempts,
        reward_threshold=reward_threshold,
        session_dir=session_dir,
        metrics_tracker=metrics_tracker,
        success_attempts_required=success_attempts_required,
        max_reflection_rules=max_reflection_rules,
        learning_strategy=learning_strategy,
        max_reflection_examples=max_reflection_examples,
    )

    while learning_coordinator.should_continue_attempts():
        attempt_timestamp = datetime.now().strftime("%H%M%S")
        attempt_dir = os.path.join(
            session_dir,
            f"attempt_{learning_coordinator.get_current_attempt()}_{attempt_timestamp}",
        )

        # 1. Set up logging for the new attempt FIRST.
        # The session_dir remains constant, while the attempt_dir is unique.
        setup_logging(attempt_dir=attempt_dir, session_dir=session_dir)

        # 2. NOW create the agent instance. Its __init__ logs will be captured.
        agent = CybORGAgentCoordinator(log_summary=False)
        agent_name = agent.__class__.__name__

        # Add logging to match the evaluation run structure
        logger.info(
            f"--- Starting Continual Learning Attempt #{learning_coordinator.get_current_attempt()} ---"
        )
        logger.info(f"Running attempt with {agent_name}...")
        logger.info(
            f"Starting Attempt {learning_coordinator.get_current_attempt()}/{learning_coordinator.max_attempts}"
        )

        # Start metrics tracking for this attempt
        metrics_tracker.start_attempt(
            learning_coordinator.get_current_attempt(), max_steps=steps
        )

        total_reward, episode_successful, failed_step, penalty_reward = (
            run_single_episode(
                cyborg,
                agent,
                episode_num=learning_coordinator.get_current_attempt(),
                steps=steps,
                is_learning_mode=True,
                reward_threshold=learning_coordinator.reward_threshold,
                metrics_tracker=metrics_tracker,
                online_learning_manager=online_learning_manager,
                online_learning_args={
                    "agents_to_improve": agents_to_improve,
                    "helper_agents": helper_agents,
                    "agent_log_map": agent_log_map,
                    "max_reflection_rules": max_reflection_rules,
                },
                learning_on_step=learning_on_step,
                learning_off_step=learning_off_step,
                max_reflection_rules=max_reflection_rules,
                disable_trajectory_pruning=disable_trajectory_pruning,
                disable_system_prompt_pruning=disable_system_prompt_pruning,
            )
        )

        # Complete metrics tracking for this attempt
        metrics_tracker.complete_attempt(
            learning_coordinator.get_current_attempt(),
            success=episode_successful,
            failure_reason=None if episode_successful else "reflection_triggered",
        )

        # Report the result to the coordinator and let it decide the next step
        learning_coordinator.report_attempt_result(episode_successful)

        if not episode_successful:
            # Trigger the learning process only on failure
            snapshot = agent.create_learning_snapshot(
                failed_step=failed_step,
                reward=penalty_reward,
                total_reward=total_reward,
                reward_threshold=reward_threshold,
                agents_to_improve=agents_to_improve,
                helper_agents=helper_agents,
                agent_log_map=agent_log_map,
                max_reflection_rules=max_reflection_rules,
                disable_trajectory_pruning=disable_trajectory_pruning,
                disable_system_prompt_pruning=disable_system_prompt_pruning,
                session_learning_history=learning_coordinator.session_learning_history,
            )
            learning_coordinator.learn_from_failure(snapshot)

        # Advance the attempt counter only after all processing for the current attempt is done
        learning_coordinator.advance_attempt()

    # Log final session summary
    session_summary = metrics_tracker.get_session_summary()
    logger.info("Continual learning session complete.")
    logger.info("===== LEARNING SESSION SUMMARY =====")
    logger.info(f"Total Attempts: {session_summary['total_attempts']}")
    logger.info(f"Successful Attempts: {session_summary['successful_attempts']}")
    logger.info(f"Success Rate: {session_summary['success_rate']:.2%}")
    logger.info(
        f"Average Steps per Attempt: {session_summary['average_steps_per_attempt']:.1f}"
    )
    logger.info(f"Best Reward: {session_summary['best_reward']:.2f}")
    logger.info(f"Average Reward: {session_summary['average_reward']:.2f}")
    logger.info(f"Reflections Triggered: {session_summary['reflections_triggered']}")
    logger.info(f"Metrics saved to: {metrics_tracker.metrics_file}")
    
    # Save the easy-to-parse session results
    learning_coordinator.save_session_results()

    # Run final evaluation to test learned heuristics
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING FINAL EVALUATION WITH LEARNED HEURISTICS")
    logger.info("=" * 60)

    # Create new agent for final evaluation (will load updated heuristics)
    final_eval_agent = CybORGAgentCoordinator(log_summary=True)

    # Run final evaluation episode
    final_total_reward, final_successful, _, _ = run_single_episode(
        cyborg,
        final_eval_agent,
        episode_num=999,
        steps=steps,
        is_learning_mode=False,
        metrics_tracker=None,
        disable_trajectory_pruning=disable_trajectory_pruning,
        disable_system_prompt_pruning=disable_system_prompt_pruning,
    )

    # Log final evaluation results
    logger.info("===== FINAL EVALUATION RESULTS =====")
    logger.info(f"Final Evaluation Reward: {final_total_reward:.2f}")
    logger.info(f"Final Evaluation Success: {'YES' if final_successful else 'NO'}")

    # Compare with learning session performance
    if session_summary["best_reward"] > 0:
        improvement = final_total_reward - session_summary["best_reward"]
        logger.info(f"Improvement over best learning attempt: {improvement:+.2f}")

    logger.info("=" * 60)
    logger.info("CONTINUAL LEARNING AND EVALUATION COMPLETE")


def run_single_episode(
    cyborg: CybORG,
    agent: CybORGAgentCoordinator,
    episode_num: int,
    steps: int,
    is_learning_mode: bool = False,
    reward_threshold: float = -0.1,
    metrics_tracker: LearningMetricsTracker = None,
    online_learning_manager=None,
    online_learning_args: dict = None,
    learning_on_step: int = 1,
    learning_off_step: int = -1,
    max_reflection_rules: int = 100,
    disable_trajectory_pruning: bool = False,
    disable_system_prompt_pruning: bool = False,
):
    total_reward = 0
    observation = cyborg.reset()
    action_space = cyborg.get_action_space("Blue")
    agent.set_initial_values(action_space, observation.observation)
    wrapped_env = cyborg
    reward = 0
    penalty_reward = 0
    episode_successful = True
    failed_step = -1
    logger = logging.getLogger(__name__)

    for step in range(steps):
        attempt_num_str_start = (
            f"{50 * ' ⬇️'}: (Attempt #{episode_num})" if is_learning_mode else ""
        )
        print("\n\n\n\n\n")
        print("=" * 120)
        print(f" CAGE STEP {step + 1}/{steps}{attempt_num_str_start} START ")
        print("=" * 120)

        action = agent.get_action(observation, action_space, reward, False)
        results = wrapped_env.step("Blue", action)
        total_reward += results.reward
        reward = results.reward

        # Log step metrics if we have a metrics tracker
        if metrics_tracker and is_learning_mode:
            metrics_tracker.log_step(
                attempt_number=episode_num,
                step=step + 1,
                action=str(action),
                reward=results.reward,
            )

        if is_learning_mode:
            # Resolve default for learning_off_step if not provided
            effective_learning_off_step = (
                learning_off_step if learning_off_step > 0 else steps
            )

            logger.info(
                f"MAIN_LOOP| Attempt={episode_num} | Step={step+1} | Action={action} | Reward={results.reward} | TotalReward={total_reward} \n\n\n"
            )
            # Check reward threshold and trigger reflection if needed
            if (
                results.reward < reward_threshold
                and learning_on_step <= (step + 1) <= effective_learning_off_step
            ):
                if online_learning_manager:
                    logger.warning(
                        f"Reward {results.reward} is below threshold {reward_threshold} within learning step window. Triggering ONLINE reflection."
                    )
                    snapshot = agent.create_learning_snapshot(
                        failed_step=step + 1,
                        reward=results.reward,
                        total_reward=total_reward,
                        reward_threshold=reward_threshold,
                        agents_to_improve=online_learning_args["agents_to_improve"],
                        helper_agents=online_learning_args["helper_agents"],
                        agent_log_map=online_learning_args["agent_log_map"],
                        max_reflection_rules=online_learning_args[
                            "max_reflection_rules"
                        ],
                        disable_trajectory_pruning=disable_trajectory_pruning,
                        disable_system_prompt_pruning=disable_system_prompt_pruning,
                        session_learning_history=[],
                    )
                    online_learning_manager.submit_failure_snapshot(snapshot)

                else:
                    print(f"\n{50 * '❌'}\n")
                    logger.warning(
                        f"Reward {results.reward} is below threshold {reward_threshold} within learning step window. Ending attempt to trigger OFFLINE reflection."
                    )
                    print(f"\n{50 * '❌'}\n")

                    # Record the specific penalty reward that caused the failure
                    penalty_reward = results.reward

                    # Finalize trajectories before reflection
                    logger.info("Finalizing trajectories before reflection...")
                    agent.get_action(
                        results,  # Pass the results as the new observation
                        action_space,
                        results.reward,
                        debug=False,
                        finalize=True,
                    )

                    # Log reflection in metrics
                    if metrics_tracker:
                        metrics_tracker.log_reflection(
                            attempt_number=episode_num,
                            step=step + 1,
                            reason=f"reward_{results.reward}_below_threshold_{reward_threshold}",
                        )

                    failed_step = step + 1
                    episode_successful = False
                    break
            elif results.reward < reward_threshold:
                logger.warning(
                    f"Reward {results.reward} is below threshold {reward_threshold}, but step {step+1} is outside the learning window ({learning_on_step}-{effective_learning_off_step}). No reflection will be triggered."
                )

        else:
            logger.info(
                f"MAIN_LOOP| Episode={episode_num} | Step={step+1} | Action={action} | Reward={results.reward} | TotalReward={total_reward}"
            )

        attempt_num_str_end = (
            f"{50 * ' ⬆️'}: (Attempt #{episode_num})" if is_learning_mode else ""
        )
        print(
            f"============== CAGE STEP {step + 1}/{steps}{attempt_num_str_end} END ==============\n"
        )
        observation = results

    agent.end_episode(total_reward, episode_num)
    return total_reward, episode_successful, failed_step, penalty_reward


if __name__ == "__main__":
    main()
