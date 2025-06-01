from Bio import Entrez
from typing import List, Dict, Any

# IMPORTANT: NCBI requires you to identify yourself. 
# Replace with a valid email address for your application.
Entrez.email = "your_app@example.com" 

async def search_pubmed(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Searches PubMed for articles matching the query and fetches basic details.

    Args:
        query: The search term (e.g., "cancer immunotherapy").
        max_results: The maximum number of results to return.

    Returns:
        A list of dictionaries, each containing details of a PubMed article.
        Returns an empty list if an error occurs during the search or fetch.
    """
    results = []
    try:
        print(f"Searching PubMed with query: '{query}', max_results: {max_results}")
        # 1. Search PubMed to get PMIDs
        handle = Entrez.esearch(db="pubmed", term=query, retmax=str(max_results), sort="relevance")
        search_results = Entrez.read(handle)
        handle.close()
        pmids = search_results["IdList"]

        if not pmids:
            print("No PMIDs found for the query.")
            return []

        print(f"Found {len(pmids)} PMIDs. Fetching summaries...")
        # 2. Fetch summaries for the PMIDs
        handle = Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        # Check if records['PubmedArticle'] exists and is iterable
        if 'PubmedArticle' in records and isinstance(records['PubmedArticle'], list):
            for record in records['PubmedArticle']:
                try:
                    article = record['MedlineCitation']['Article']
                    pmid = record['MedlineCitation']['PMID']
                    title = article.get('ArticleTitle', 'No Title Available')
                    abstract_dict = article.get('Abstract', {})
                    # Handle cases where AbstractText is a list or a single string
                    abstract_parts = abstract_dict.get('AbstractText', [])
                    if isinstance(abstract_parts, list):
                        abstract = "\n".join(abstract_parts)
                    else:
                        abstract = str(abstract_parts) # Convert non-list to string
                        
                    # Try to get authors (handle potential KeyError)
                    authors_list = article.get('AuthorList', [])
                    authors = ", ".join([
                        f"{author.get('LastName', '')} {author.get('Initials', '')}"
                        for author in authors_list if isinstance(author, dict)
                    ]) if authors_list else "No Authors Listed"

                    results.append({
                        "id": str(pmid),
                        "title": str(title),
                        "abstract": str(abstract) if abstract else "No Abstract Available",
                        "authors": authors,
                        "source": "PubMed"
                    })
                except Exception as e:
                    print(f"Error parsing PubMed record (PMID maybe?): {e} - Skipping record.")
                    # Optionally log the problematic record here
                    continue # Skip this record and continue with the next
        else:
            print("Warning: 'PubmedArticle' key not found or not a list in Entrez result.")
            print(f"Records structure: {records.keys()}") # Log keys to understand structure

        print(f"Successfully parsed {len(results)} PubMed records.")

    except Exception as e:
        print(f"Error during Entrez operation: {e}")
        # Return empty list on error to avoid breaking the API endpoint
        return []

    return results

# Example Usage (for testing this module directly)
if __name__ == '__main__':
    import asyncio

    async def test_search():
        # test_query = "mRNA vaccine cancer"
        test_query = "(CAR-T) AND (Glioblastoma)"
        print(f"--- Testing PubMed Search with query: '{test_query}' ---")
        search_results = await search_pubmed(test_query, max_results=5)
        
        if search_results:
            print(f"\n--- Found {len(search_results)} Results ---")
            for i, result in enumerate(search_results):
                print(f"\nResult {i+1}:")
                print(f"  PMID: {result['id']}")
                print(f"  Title: {result['title']}")
                print(f"  Authors: {result['authors']}")
                print(f"  Abstract: {result['abstract'][:200]}...") # Print first 200 chars
        else:
            print("\n--- No results returned from search_pubmed ---")

    asyncio.run(test_search())