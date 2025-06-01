import os
import json
import hashlib
from typing import Union, Tuple
from web3 import Web3
# Comment out the problematic import, as it's not needed for Hardhat Network
# from web3.middleware import geth_poa_middleware # For some dev networks
from dotenv import load_dotenv

# Load environment variables from .env file located at the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env') 
load_dotenv(dotenv_path=dotenv_path)

# --- Configuration --- 
RPC_URL = os.getenv("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545")
CONTRACT_ADDRESS = os.getenv("CONTRIBUTION_TRACKER_ADDRESS", "0x5FbDB2315678afecb367f032d93F642f64180aa3")
# IMPORTANT: Use the private key of the account that deployed the contract (usually Account #0 from Hardhat node)
# Store this securely in your .env file, DO NOT commit it directly.
# Example for Hardhat node Account #0:
# BLOCKCHAIN_PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
OWNER_PRIVATE_KEY = os.getenv("BLOCKCHAIN_PRIVATE_KEY") 

# Path to the contract ABI JSON file (adjust relative path as needed)
ABI_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'blockchain', 'artifacts', 'contracts', 'ContributionTracker.sol', 'ContributionTracker.json')

# --- Web3 Connection and Contract Setup --- 
w3 = None
contract = None
owner_account = None

try:
    # Connect to the blockchain node
    w3 = Web3(Web3.HTTPProvider(RPC_URL))

    # Inject PoA middleware if needed (common for dev nets like Goerli, Sepolia, sometimes Hardhat)
    # w3.middleware_onion.inject(geth_poa_middleware, layer=0)

    if not w3.is_connected():
        raise ConnectionError(f"Failed to connect to blockchain node at {RPC_URL}")

    print(f"Connected to blockchain: Chain ID {w3.eth.chain_id}")

    # Load Contract ABI
    try:
        with open(ABI_PATH, 'r') as f:
            contract_interface = json.load(f)
        contract_abi = contract_interface['abi']
    except FileNotFoundError:
        raise FileNotFoundError(f"Contract ABI file not found at: {ABI_PATH}. Ensure the contract is compiled.")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON ABI file: {ABI_PATH}")
    except KeyError:
        raise ValueError(f"'abi' key not found in JSON file: {ABI_PATH}")

    # Get contract instance
    checksum_address = w3.to_checksum_address(CONTRACT_ADDRESS)
    contract = w3.eth.contract(address=checksum_address, abi=contract_abi)

    # Load owner account from private key
    if not OWNER_PRIVATE_KEY:
         print("WARNING: BLOCKCHAIN_PRIVATE_KEY not set in .env. Blockchain logging will fail.")
         # raise ValueError("BLOCKCHAIN_PRIVATE_KEY environment variable not set.")
    else:
        owner_account = w3.eth.account.from_key(OWNER_PRIVATE_KEY)
        print(f"Blockchain utility initialized. Using owner account: {owner_account.address}")

except Exception as e:
    print(f"ERROR initializing blockchain utility: {e}")
    # Allow app to continue but log error; subsequent calls will fail gracefully
    w3 = None 
    contract = None
    owner_account = None

# --- Core Function --- 
async def record_contribution(contribution_type: str, data_to_log: str) -> Tuple[bool, Union[str, None]]:
    """
    Calculates a hash of the data and logs metadata to the ContributionTracker contract.

    Args:
        contribution_type: The type of contribution (e.g., "AI_Feedback").
        data_to_log: The string data whose hash should be logged (e.g., feedback text + context).

    Returns:
        A tuple: (success_boolean, transaction_hash_or_error_message)
    """
    if not w3 or not contract or not owner_account:
        error_msg = "Blockchain connection/contract not initialized properly."
        print(f"ERROR: record_contribution failed - {error_msg}")
        return False, error_msg

    try:
        # 1. Calculate SHA-256 hash of the input data
        # Ensure data is encoded to bytes before hashing
        data_bytes = data_to_log.encode('utf-8')
        reference_hash_bytes = hashlib.sha256(data_bytes).digest() # Use .digest() for bytes32
        
        print(f"Recording contribution: Type='{contribution_type}', Hash={reference_hash_bytes.hex()}")

        # 2. Build the transaction to call logContribution
        nonce = w3.eth.get_transaction_count(owner_account.address)
        
        # Estimate gas (optional but recommended)
        try:
            gas_estimate = contract.functions.logContribution(
                contribution_type, 
                reference_hash_bytes
            ).estimate_gas({'from': owner_account.address})
        except Exception as estimate_err:
             print(f"Warning: Gas estimation failed ({estimate_err}). Using default gas limit.")
             gas_estimate = 300000 # Fallback gas limit

        tx_params = {
            'from': owner_account.address,
            'nonce': nonce,
            'gas': gas_estimate,
            # Let web3.py determine gas price for local dev network
            # 'gasPrice': w3.eth.gas_price, 
            # For EIP-1559 networks (like newer Ethereum testnets/mainnet):
            # 'maxFeePerGas': w3.to_wei('2', 'gwei'),
            # 'maxPriorityFeePerGas': w3.to_wei('1', 'gwei'),
        }

        transaction = contract.functions.logContribution(
            contribution_type, 
            reference_hash_bytes
        ).build_transaction(tx_params)

        # 3. Sign the transaction
        signed_tx = w3.eth.account.sign_transaction(transaction, private_key=OWNER_PRIVATE_KEY)

        # 4. Send the transaction
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        print(f"Transaction sent: {tx_hash.hex()}")

        # 5. (Optional) Wait for the transaction receipt
        # tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        # print(f"Transaction confirmed in block: {tx_receipt.blockNumber}")
        # if tx_receipt.status == 0:
        #    return False, f"Transaction failed (receipt status 0): {tx_hash.hex()}"

        return True, tx_hash.hex()

    except Exception as e:
        error_msg = f"Error sending blockchain transaction: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg

# --- Helper/Getter Functions (Optional) ---
async def get_contribution_details(contribution_id: int) -> Union[dict, None]:
    """Retrieves contribution details from the blockchain."""
    if not w3 or not contract:
        print("ERROR: get_contribution_details failed - Blockchain connection/contract not initialized.")
        return None
    try:
        result = contract.functions.getContribution(contribution_id).call()
        # Convert result tuple to a dictionary for easier use
        return {
            'id': result[0],
            'contributor': result[1],
            'contributionType': result[2],
            'referenceHash': result[3].hex(), # Convert bytes32 to hex string
            'timestamp': result[4]
        }
    except Exception as e:
        print(f"Error retrieving contribution {contribution_id}: {e}")
        return None 