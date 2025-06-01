from abc import ABC, abstractmethod
from typing import Any, Dict

class AgentInterface(ABC):
    """ Abstract Base Class for all specialized AI agents. """

    @property
    @abstractmethod
    def name(self) -> str:
        """ A unique identifier name for the agent (e.g., 'data_analyzer', 'scheduler'). """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """ A brief description of what the agent does, potentially used for selection. """
        pass
    
    @abstractmethod
    async def run(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        The main execution method for an agent.

        Args:
            context: A dictionary containing relevant context,
                     e.g., patient data, previous steps' results, conversation history.
            **kwargs: Additional arguments specific to the agent or task,
                      e.g., user prompt details, specific parameters like target date for scheduling.

        Returns:
            A dictionary containing the result of the agent's execution.
            The structure should include at least 'status' and 'output'.
            Example:
            {
                "status": "success" | "failure" | "requires_review" | "clarification_needed",
                "output": { ... agent specific output, e.g., summary text, draft notification, schedule options ... },
                "summary": "Brief summary of action taken/result for display or logging.",
                "error_message": "..." # Optional: if status is failure
            }
        """
        pass 