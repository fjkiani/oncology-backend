from abc import ABC, abstractmethod
from typing import Dict, Any

class AgentInterface(ABC):
    """Abstract base class for all agents."""

    @abstractmethod
    def __init__(self):
        """Initialize the agent, setting up any required resources or configurations."""
        self.name: str = "BaseAgent"
        self.description: str = "A base agent interface."
        pass

    @abstractmethod
    async def run(self, patient_data: Dict[str, Any], prompt_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        The main execution method for the agent.

        Args:
            patient_data: A dictionary containing relevant data for the patient.
            prompt_details: A dictionary containing the prompt or command specifics,
                              including intent, entities, or other parameters.

        Returns:
            A dictionary containing the agent's output, status, and any other relevant information.
        """
        pass 