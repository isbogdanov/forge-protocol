import logging
from base_classes.executor_base import BaseToolExecutor

class ExemplifierToolExecutor(BaseToolExecutor):
    """
    Executor for the Exemplifier agent.
    It uses the generic monologue tools to reason about the examples it generates.
    """
    def __init__(self, logger: logging.Logger, **kwargs):
        super().__init__(logger, **kwargs)
        # Explicitly expose the generic tools (raise_a_question, etc.)
        self.tools = self.generic_tools.get_tools()

