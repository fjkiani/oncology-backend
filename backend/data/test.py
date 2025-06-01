import requests
import json

# Base URL for cBioPortal API
BASE_URL = "https://www.cbioportal.org/api"

# TCGA-BRCA study and molecular profile IDs
molecular_profile_id = "brca_tcga_mutations"
sample_list_id = "brca_tcga_all"
projection = "DETAILED"

# Construct the API endpoint URL
# Try the simpler /mutations endpoint with GET
endpoint = f"/molecular-profiles/{molecular_profile_id}/mutations"
url = f"{BASE_URL}{endpoint}"

# Parameters for the request
params = {
    "sampleListId": sample_list_id,
    "projection": projection
}

# Headers for the request
headers = {
    "accept": "application/json"
}

print(f"Fetching mutations via GET from: {url}")
print(f"With parameters: {params}")

# Make the API request using GET
try:
    response = requests.get(url, headers=headers, params=params, timeout=120) # Longer timeout
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

    # Parse the JSON response
    mutations = response.json()

    # Check if we got data
    if isinstance(mutations, list):
        print(f"Successfully fetched {len(mutations)} mutations.")

        # Save the data to a JSON file
        output_filename = "brca_tcga_mutations.json"
        try:
            with open(output_filename, 'w') as f:
                json.dump(mutations, f, indent=2) # Use indent for readability
            print(f"Saved mutation data to {output_filename}")
        except IOError as e:
            print(f"Error saving data to file: {e}")

        # Print the first 5 mutations
        print("First 5 mutations:")
        for i, mut in enumerate(mutations[:5]):
            # Pretty print the JSON for readability
            print(f"Mutation {i+1}: {json.dumps(mut, indent=2)}")
    elif mutations is None:
         print("API returned None. No mutations found or error occurred.")
    else:
        print(f"Received unexpected data format: {type(mutations)}")
        print(mutations)

except requests.exceptions.Timeout as e:
    print(f"Error: The request timed out: {e}")
except requests.exceptions.HTTPError as e:
    print(f"Error: HTTP Error occurred: {e.response.status_code} {e.response.reason}")
    # Print response body if available, might contain error details from API
    try:
        print(f"Response body: {e.response.json()}")
    except json.JSONDecodeError:
        print(f"Response body (non-JSON): {e.response.text}")
except requests.exceptions.RequestException as e:
    print(f"Error: An unexpected error occurred during the request: {e}")
except json.JSONDecodeError as e:
    print(f"Error: Failed to decode JSON response: {e}")
    print(f"Raw response text: {response.text[:500]}...") # Print beginning of response
