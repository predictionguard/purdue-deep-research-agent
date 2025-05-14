from typing import Any, List, Optional
import requests
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server with working directory
mcp = FastMCP("clinicaltrials-mcp", working_dir="/home/kidzik/workspace/pubmed-mcp")

# Constants
API_BASE_URL = "https://clinicaltrials.gov/api/v2"
TOOL_NAME = "clinicaltrials-mcp"

def make_api_request(endpoint: str, params: dict) -> Any:
    """Make a request to the ClinicalTrials.gov API with proper error handling."""
    url = f"{API_BASE_URL}/{endpoint}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json"
    }
    
    try:
        print(url)
        print(params)
        print(headers)
        response = requests.get(url, params=params, headers=headers, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def search_trials(query: str, max_results: int = 10) -> str:
    """Search ClinicalTrials.gov for studies matching the query.
    
    Args:
        query: Search query 
        max_results: Maximum number of results to return (default: 10)
    """
    search_params = {
        "query.term": query,
        "pageSize": max_results,
        "format": "json"
    }
    
    search_results = make_api_request("studies", search_params)
    
    if isinstance(search_results, dict) and "error" in search_results:
        return f"Error searching ClinicalTrials.gov: {search_results['error']}"
    
    studies = search_results.get("studies", [])
    if not studies:
        return "No results found for your query."
    
    formatted_results = []
    
    for study in studies:
        protocol_section = study.get("protocolSection", {})
        identification = protocol_section.get("identificationModule", {})
        status = protocol_section.get("statusModule", {})
        
        nct_id = identification.get("nctId", "Unknown ID")
        title = identification.get("briefTitle", "No title")
        status_text = status.get("overallStatus", "Unknown status")
        phase = protocol_section.get("phaseModule", {}).get("phase", "Unknown phase")
        
        formatted_results.append(
            f"Title: {title}\n"
            f"ID: {nct_id}\n"
            f"Status: {status_text}\n"
            f"Phase: {phase}"
        )
    
    if not formatted_results:
        return "No study details could be retrieved."
        
    return "\n\n---\n\n".join(formatted_results)

@mcp.tool()
def get_trial_details(nct_id: str) -> str:
    """Get detailed information about a specific clinical trial by its NCT ID.
    
    Args:
        nct_id: The NCT identifier for the trial
    """
    study_params = {"format": "json"}
    
    study_details = make_api_request(f"studies/{nct_id}", study_params)
    
    if isinstance(study_details, dict) and "error" in study_details:
        return f"Error retrieving trial details: {study_details['error']}"
    
    protocol_section = study_details.get("protocolSection", {})
    
    # Extract main sections
    identification = protocol_section.get("identificationModule", {})
    status = protocol_section.get("statusModule", {})
    sponsor = protocol_section.get("sponsorCollaboratorsModule", {})
    design = protocol_section.get("designModule", {})
    conditions = protocol_section.get("conditionsModule", {})
    description = protocol_section.get("descriptionModule", {})
    
    # Format the details
    title = identification.get("briefTitle", "No title")
    official_title = identification.get("officialTitle", "No official title")
    status_text = status.get("overallStatus", "Unknown status")
    phase = protocol_section.get("phaseModule", {}).get("phase", "Unknown phase")
    
    primary_sponsor = sponsor.get("leadSponsor", {}).get("name", "Unknown sponsor")
    
    study_type = design.get("studyType", "Unknown type")
    primary_purpose = design.get("primaryPurpose", "Unknown purpose")
    
    condition_list = conditions.get("conditions", [])
    conditions_text = ", ".join(condition_list) if condition_list else "None specified"
    
    detailed_description = description.get("detailedDescription", "No detailed description available")
    
    formatted_details = (
        f"NCT ID: {nct_id}\n\n"
        f"Brief Title: {title}\n\n"
        f"Official Title: {official_title}\n\n"
        f"Status: {status_text}\n\n"
        f"Phase: {phase}\n\n"
        f"Sponsor: {primary_sponsor}\n\n"
        f"Study Type: {study_type}\n\n"
        f"Primary Purpose: {primary_purpose}\n\n"
        f"Conditions: {conditions_text}\n\n"
        f"Detailed Description: {detailed_description}"
    )
    
    return formatted_details

@mcp.tool()
def find_trials_by_condition(condition: str, max_results: int = 10) -> str:
    """Search for clinical trials related to a specific medical condition.
    
    Args:
        condition: Medical condition or disease
        max_results: Maximum number of results to return (default: 10)
    """
    search_params = {
        "query.cond": condition,
        "pageSize": max_results,
        "format": "json"
    }
    
    search_results = make_api_request("studies", search_params)
    
    if isinstance(search_results, dict) and "error" in search_results:
        return f"Error searching by condition: {search_results['error']}"
    
    return format_search_results(search_results)

@mcp.tool()
def find_trials_by_location(location: str, max_results: int = 10) -> str:
    """Search for clinical trials in a specific location.
    
    Args:
        location: Location (city, state, country)
        max_results: Maximum number of results to return (default: 10)
    """
    search_params = {
        "query.locn": location,
        "pageSize": max_results,
        "format": "json"
    }
    
    search_results = make_api_request("studies", search_params)
    
    if isinstance(search_results, dict) and "error" in search_results:
        return f"Error searching by location: {search_results['error']}"
    
    return format_search_results(search_results)

def format_search_results(search_results: dict) -> str:
    """Helper function to format search results."""
    studies = search_results.get("studies", [])
    if not studies:
        return "No results found for your query."
    
    formatted_results = []
    
    for study in studies:
        protocol_section = study.get("protocolSection", {})
        identification = protocol_section.get("identificationModule", {})
        status = protocol_section.get("statusModule", {})
        
        nct_id = identification.get("nctId", "Unknown ID")
        title = identification.get("briefTitle", "No title")
        status_text = status.get("overallStatus", "Unknown status")
        phase = protocol_section.get("phaseModule", {}).get("phase", "Unknown phase")
        
        formatted_results.append(
            f"Title: {title}\n"
            f"ID: {nct_id}\n"
            f"Status: {status_text}\n"
            f"Phase: {phase}"
        )
    
    if not formatted_results:
        return "No study details could be retrieved."
        
    return "\n\n---\n\n".join(formatted_results)

if __name__ == "__main__":
    mcp.run(transport='stdio')
