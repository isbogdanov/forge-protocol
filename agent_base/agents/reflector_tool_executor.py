import logging
from base_classes.executor_base import BaseToolExecutor


class ReflectorToolExecutor(BaseToolExecutor):
    def __init__(self, logger: logging.Logger, **kwargs):
        super().__init__(logger, **kwargs)
        # The Reflector agent only uses the generic monologue tools
        self.tools = self.generic_tools.get_tools()
