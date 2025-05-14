from typing import Any, List, Optional
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server with working directory
mcp = FastMCP("pubmed-mcp", working_dir="/home/kidzik/workspace/pubmed-mcp")

# Constants
ENTREZ_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DATABASE = "pubmed"
TOOL_NAME = "pubmed-mcp"
EMAIL = "sharan@predictionguard.com"  # Replace with your email

async def make_entrez_request(endpoint: str, params: dict, is_json: bool = True) -> Any:
    """Make a request to the Entrez API with proper error handling."""
    url = f"{ENTREZ_BASE_URL}/{endpoint}.fcgi"
    
    # Add required parameters
    params.update({
        "db": DATABASE,
        "tool": TOOL_NAME,
        "email": EMAIL,
    })
    
    if is_json:
        params["retmode"] = "json"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=30.0)
            response.raise_for_status()
            
            if is_json:
                return response.json()
            return response.text
        except Exception as e:
            return {"error": str(e)} if is_json else f"Error: {str(e)}"

@mcp.tool()
async def search_pubmed(query: str, max_results: int = 10) -> str:
    """Search PubMed for articles matching the query.
    
    Args:
        query: Search query in PubMed syntax
        max_results: Maximum number of results to return (default: 10)
    """
    # First use ESearch to get IDs
    search_params = {
        "term": query,
        "retmax": max_results,
    }
    search_results = await make_entrez_request("esearch", search_params)
    
    if isinstance(search_results, dict) and "error" in search_results:
        return f"Error searching PubMed: {search_results['error']}"
    
    id_list = search_results.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return "No results found for your query."
    
    # Use ESummary to get summaries for these IDs
    id_param = ",".join(id_list)
    summary_params = {
        "id": id_param,
    }
    summary_results = await make_entrez_request("esummary", summary_params)
    
    if isinstance(summary_results, dict) and "error" in summary_results:
        return f"Error fetching article summaries: {summary_results['error']}"
    
    # Format results
    formatted_results = []
    result_data = summary_results.get("result", {})
    
    for article_id in id_list:
        article = result_data.get(article_id, {})
        if not article:
            continue
            
        title = article.get("title", "No title")
        authors = ", ".join([a.get("name", "") for a in article.get("authors", [])])
        if not authors:
            authors = "No authors listed"
            
        pubdate = article.get("pubdate", "Unknown date")
        journal = article.get("source", "Unknown journal")
        
        formatted_results.append(
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Published: {pubdate} in {journal}\n"
            f"PMID: {article_id}"
        )
    
    if not formatted_results:
        return "No article details could be retrieved."
        
    return "\n\n---\n\n".join(formatted_results)

@mcp.tool()
async def get_pubmed_abstract(pmid: str) -> str:
    """Get the abstract for a specific PubMed article by its PMID.
    
    Args:
        pmid: PubMed ID of the article
    """
    # Use EFetch to retrieve the full abstract
    fetch_params = {
        "id": pmid,
        "rettype": "abstract",
    }
    
    # For abstracts, we need plain text
    abstract_text = await make_entrez_request("efetch", fetch_params, is_json=False)
    
    if abstract_text.startswith("Error:"):
        return abstract_text
        
    if not abstract_text.strip():
        return "No abstract available for this article."
        
    return abstract_text

@mcp.tool()
async def get_related_articles(pmid: str, max_results: int = 5) -> str:
    """Find articles related to a specific PubMed article.
    
    Args:
        pmid: PubMed ID of the seed article
        max_results: Maximum number of related articles to return (default: 5)
    """
    # Use ELink to find related articles
    link_params = {
        "id": pmid,
        "dbfrom": DATABASE,
        "cmd": "neighbor_score",
        "linkname": "pubmed_pubmed"
    }
    
    link_results = await make_entrez_request("elink", link_params)
    
    if isinstance(link_results, dict) and "error" in link_results:
        return f"Error finding related articles: {link_results['error']}"
    
    # Extract related article IDs
    related_ids = []
    
    try:
        linksets = link_results.get("linksets", [])
        if not linksets:
            return "No related articles found."
            
        for linkset in linksets:
            for linksetdb in linkset.get("linksetdbs", []):
                if linksetdb.get("linkname") == "pubmed_pubmed":
                    related_ids = [str(link) for link in linksetdb.get("links", [])][:max_results]
                    break
        
        if not related_ids:
            return "No related articles found."
    except Exception as e:
        return f"Error processing related articles data: {str(e)}"
    
    # Get summaries for related articles
    summary_params = {
        "id": ",".join(related_ids),
    }
    summary_results = await make_entrez_request("esummary", summary_params)
    
    if isinstance(summary_results, dict) and "error" in summary_results:
        return f"Error fetching related article details: {summary_results['error']}"
    
    # Format results
    formatted_results = []
    result_data = summary_results.get("result", {})
    
    for article_id in related_ids:
        article = result_data.get(article_id, {})
        if not article:
            continue
            
        title = article.get("title", "No title")
        authors = ", ".join([a.get("name", "") for a in article.get("authors", [])])
        if not authors:
            authors = "No authors listed"
            
        pubdate = article.get("pubdate", "Unknown date")
        
        formatted_results.append(
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Published: {pubdate}\n"
            f"PMID: {article_id}"
        )
    
    if not formatted_results:
        return "No related article details could be retrieved."
        
    return "\n\n---\n\n".join(formatted_results)

@mcp.tool()
async def find_by_author(author: str, max_results: int = 10) -> str:
    """Search PubMed for articles by a specific author.
    
    Args:
        author: Author name (e.g., "Smith JB")
        max_results: Maximum number of results to return (default: 10)
    """
    # Construct author query
    author_query = f"{author}[Author]"
    
    # Use the existing search_pubmed function
    return await search_pubmed(author_query, max_results)

if __name__ == "__main__":
    # Initialize and run the server with working directory

    mcp.run(transport='stdio') 