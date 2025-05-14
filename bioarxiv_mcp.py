from typing import Any, List, Optional
import httpx
import argparse
from mcp.server.fastmcp import FastMCP

def parse_args():
    parser = argparse.ArgumentParser(description='BioRxiv/MedRxiv MCP Service')
    parser.add_argument('--server', choices=['biorxiv', 'medrxiv'], default='biorxiv',
                        help='Select server: biorxiv or medrxiv (default: biorxiv)')
    parser.add_argument('--working-dir', default="/home/kidzik/workspace/pubmed-mcp",
                        help='Working directory for MCP')
    return parser.parse_args()

# Initialize FastMCP server with working directory
mcp = FastMCP("biorxiv-mcp", working_dir="/home/kidzik/workspace/pubmed-mcp")

# Constants
API_BASE_URL = "https://api.biorxiv.org"
TOOL_NAME = "biorxiv-mcp"

async def make_api_request(endpoint: str, params: dict = None) -> Any:
    """Make a request to the bioRxiv API with proper error handling."""
    url = f"{API_BASE_URL}/{endpoint}"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

@mcp.tool()
async def get_preprint_by_doi(server: str, doi: str) -> str:
    """Get detailed information about a specific preprint by its DOI.
    
    Args:
        server: Server to search ("biorxiv" or "medrxiv")
        doi: DOI of the preprint (e.g., "10.1101/2020.01.01.123456")
    """
    endpoint = f"details/{server}/{doi}/na/json"
    
    results = await make_api_request(endpoint)
    
    if isinstance(results, dict) and "error" in results:
        return f"Error retrieving preprint details: {results['error']}"
    
    collection = results.get("collection", [])
    if not collection or len(collection) == 0:
        return f"No preprint found with DOI: {doi}"
    
    preprint = collection[0]
    doi = preprint.get("doi", "Unknown DOI")
    title = preprint.get("title", "No title")
    authors = preprint.get("authors", "No authors listed")
    date = preprint.get("date", "Unknown date")
    category = preprint.get("category", "Unknown category")
    abstract = preprint.get("abstract", "No abstract available")
    license = preprint.get("license", "Unknown license")
    author_corresponding = preprint.get("author_corresponding", "Unknown")
    author_corresponding_institution = preprint.get("author_corresponding_institution", "Unknown")
    
    formatted_details = (
        f"Title: {title}\n\n"
        f"Authors: {authors}\n\n"
        f"DOI: {doi}\n\n"
        f"Date: {date}\n\n"
        f"Category: {category}\n\n"
        f"License: {license}\n\n"
        f"Corresponding Author: {author_corresponding}\n\n"
        f"Institution: {author_corresponding_institution}\n\n"
        f"Abstract: {abstract}"
    )
    
    return formatted_details

@mcp.tool()
async def find_published_version(server: str, doi: str) -> str:
    """Find the published version of a preprint by its DOI.
    
    Args:
        server: Server to search ("biorxiv" or "medrxiv")
        doi: DOI of the preprint (e.g., "10.1101/2020.01.01.123456")
    """
    endpoint = f"pubs/{server}/{doi}/na/json"
    
    results = await make_api_request(endpoint)
    
    if isinstance(results, dict) and "error" in results:
        return f"Error finding published version: {results['error']}"
    
    collection = results.get("collection", [])
    if not collection or len(collection) == 0:
        return f"No published version found for preprint with DOI: {doi}"
    
    publication = collection[0]
    preprint_doi = publication.get("biorxiv_doi", "Unknown preprint DOI")
    published_doi = publication.get("published_doi", "Unknown published DOI")
    published_journal = publication.get("published_journal", "Unknown journal")
    preprint_title = publication.get("preprint_title", "No title")
    preprint_date = publication.get("preprint_date", "Unknown preprint date")
    published_date = publication.get("published_date", "Unknown publication date")
    
    formatted_details = (
        f"Preprint Title: {preprint_title}\n\n"
        f"Preprint DOI: {preprint_doi}\n\n"
        f"Preprint Date: {preprint_date}\n\n"
        f"Published DOI: {published_doi}\n\n"
        f"Journal: {published_journal}\n\n"
        f"Publication Date: {published_date}"
    )
    
    return formatted_details

@mcp.tool()
async def get_recent_preprints(server: str, days: int = 7, max_results: int = 10, category: str = None) -> str:
    """Get recent preprints from bioRxiv/medRxiv.
    
    Args:
        server: Server to search ("biorxiv" or "medrxiv")
        days: Number of days to look back (default: 7)
        max_results: Maximum number of results to return (default: 10)
        category: Category to search (e.g., "cell_biology")
    """
    endpoint = f"details/{server}/{days}d/0"
    params = {}
    if category:
        params = {"category": category}
    
    results = await make_api_request(endpoint, params)
    
    if isinstance(results, dict) and "error" in results:
        return f"Error retrieving recent preprints: {results['error']}"
    
    collection = results.get("collection", [])
    if not collection:
        return f"No recent preprints found in the last {days} days."
    
    # Limit the number of results
    collection = collection[:max_results]
    
    formatted_results = []
    for preprint in collection:
        doi = preprint.get("doi", "Unknown DOI")
        title = preprint.get("title", "No title")
        authors = preprint.get("authors", "No authors listed")
        date = preprint.get("date", "Unknown date")
        category = preprint.get("category", "Unknown category")
        
        formatted_results.append(
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"DOI: {doi}\n"
            f"Date: {date}\n"
            f"Category: {category}"
        )
    
    if not formatted_results:
        return "No preprint details could be retrieved."
        
    return "\n\n---\n\n".join(formatted_results)

@mcp.tool()
async def search_preprints(server: str, start_date: str, end_date: str, max_results: int = 10, category: str = None) -> str:
    """Search for preprints in a specific category.
    
    Args:
        server: Server to search ("biorxiv" or "medrxiv")
        category: Category to search (e.g., "cell_biology")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_results: Maximum number of results to return (default: 10)
    """
    endpoint = f"details/{server}/{start_date}/{end_date}/0"
    params = {}
    if category:
        params = {"category": category}
    
    results = await make_api_request(endpoint, params)
    
    if isinstance(results, dict) and "error" in results:
        return f"Error searching by category: {results['error']}"
    
    collection = results.get("collection", [])
    if not collection:
        return f"No preprints found in category '{category}'."
    
    # Limit the number of results
    collection = collection[:max_results]
    
    formatted_results = []
    for preprint in collection:
        doi = preprint.get("doi", "Unknown DOI")
        title = preprint.get("title", "No title")
        authors = preprint.get("authors", "No authors listed")
        date = preprint.get("date", "Unknown date")
        
        formatted_results.append(
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"DOI: {doi}\n"
            f"Date: {date}"
        )
    
    if not formatted_results:
        return "No preprint details could be retrieved."
        
    return "\n\n---\n\n".join(formatted_results)

if __name__ == "__main__":
    args = parse_args()
    tool_name = f"{args.server}-mcp"
    mcp = FastMCP(tool_name, working_dir=args.working_dir)
    
    # Register all the tools defined above
    mcp.tool()(search_preprints)
    mcp.tool()(get_preprint_by_doi)
    mcp.tool()(find_published_version)
    mcp.tool()(get_recent_preprints)
    
    mcp.run(transport='stdio')