from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
import asyncio
import sqlite3

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

    def _fetch_trial_details(self, nct_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetches full trial details from SQLite for a list of NCT IDs.
        """
        if not nct_ids:
            return []
        
        conn = self._get_db_connection()
        if not conn:
            return []
            
        try:
            cursor = conn.cursor()
            placeholders = ','.join('?' for _ in nct_ids)
            query = f"SELECT * FROM trials WHERE id IN ({placeholders})"
            cursor.execute(query, nct_ids)
            
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            results_dict = {row['id']: row for row in results}
            ordered_results = [results_dict[nct_id] for nct_id in nct_ids if nct_id in results_dict]
            
            return ordered_results
        except sqlite3.Error as e:
            logging.error(f"SQLite query failed: {e}", exc_info=True)
            return []
        finally:
            conn.close() 