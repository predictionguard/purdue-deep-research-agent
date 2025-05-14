from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, Dict, Any, List
import asyncio
import sys
import os
import json
from predictionguard import PredictionGuard

# Import the MCP servers
from pubmed_mcp import search_pubmed, get_pubmed_abstract, get_related_articles, find_by_author
from clinicaltrialsgov_mcp import search_trials, get_trial_details, find_trials_by_condition, find_trials_by_location
from bioarxiv_mcp import get_preprint_by_doi, find_published_version, get_recent_preprints, search_preprints

app = FastAPI(title="Biomedical MCP API")

# Initialize PredictionGuard client
client = PredictionGuard(url=os.getenv("PREDICTIONGUARD_URL","https://api.predictionguard.com"))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to classify query using LLM
async def classify_query(query: str) -> Dict[str, Any]:
    system_prompt = """You are a biomedical query classifier.. Your task is to:
1. Select ALL relevant biomedical databases for answering the query
2. Extract any specific identifiers (PMID, DOI, NCT ID, etc.)
3. Determine the query type based on the user's request

Available databases and their specific capabilities:

PubMed MCP Server:
- Search PubMed for articles matching a query (query_type: "search")
- Retrieve abstracts for specific articles (query_type: "abstract")
- Find related articles based on a PMID (query_type: "related")
- Search for articles by a specific author (query_type: "author")

BioRxiv/MedRxiv MCP Server:
- Get detailed information about preprints by DOI (query_type: "preprint")
- Find published versions of preprints (query_type: "published")
- Search for recent preprints (query_type: "search")
- Search preprints by date range and category (query_type: "search")

ClinicalTrials.gov MCP Server:
- Search for trials matching specific criteria (query_type: "search")
- Get detailed information about specific trials (query_type: "trial")
- Find trials by medical condition (query_type: "condition")
- Find trials by location (query_type: "location")

Choose ALL databases that could provide relevant information for the query. For example:
- If query is about clinical trials and related research → PubMed AND ClinicalTrials.gov
- If query is about preprints and their published versions → BioRxiv AND PubMed
- If query is about a specific trial and related research → ClinicalTrials.gov AND PubMed

IMPORTANT: Respond with ONLY a valid JSON object, no other text. The JSON must be properly formatted with double quotes and no trailing commas.

Example responses:
1. For a search query:
{
    "databases": ["pubmed", "clinicaltrials"],
    "identifiers": {},
    "query_type": "search"
}

2. For finding related articles:
{
    "databases": ["pubmed"],
    "identifiers": {
        "pmid": "12345678"
    },
    "query_type": "related"
}

3. For auth)
search:
{
    "databases": ["pubmed"],
    "identifiers": {},
    "query_type": "author"
}

4. For finding published version of preprint:
{
    "databases": ["biorxiv"],
    "identifiers": {
        "doi": "10.1101/2023.01.01.123456"
    },
    "query_type": "published"
}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    try:
        result = client.chat.completions.create(
            model=os.getenv("PREDICTIONGUARD_MODEL","Hermes-3-Llama-3.1-70B"),
            messages=messages
        )
        
        # Parse the response - handle PredictionGuard response format
        response_text = result['choices'][0]['message']['content'].strip()
        
        # Clean up the response if needed
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        response = json.loads(response_text)
        return response
    except Exception as e:
        print(f"Error in query classification: {str(e)}")
        return {
            "databases": ["pubmed"], 
            "identifiers": {},
            "query_type": "search"
        }

# Helper function to process query for a specific database
async def process_database_query(database: str, query_info: Dict[str, Any], max_results: int) -> Dict[str, Any]:
    try:
        if database == "pubmed":
            if "pmid" in query_info["identifiers"]:
                if query_info["query_type"] == "related":
                    result = await get_related_articles(query_info["identifiers"]["pmid"], max_results)
                else:
                    result = await get_pubmed_abstract(query_info["identifiers"]["pmid"])
            elif query_info["query_type"] == "author":
                result = await find_by_author(query_info["query"], max_results)
            else:
                result = await search_pubmed(query_info["query"], max_results)
            return {"source": "pubmed", "data": result}
            
        elif database == "clinicaltrials":
            if "nct_id" in query_info["identifiers"]:
                result = get_trial_details(query_info["identifiers"]["nct_id"])
            elif query_info["query_type"] == "condition":
                result = find_trials_by_condition(query_info["query"], max_results)
            elif query_info["query_type"] == "location":
                result = find_trials_by_location(query_info["query"], max_results)
            else:
                result = search_trials(query_info["query"], max_results)
            return {"source": "clinicaltrials", "data": result}
            
        elif database == "biorxiv":
            if "doi" in query_info["identifiers"]:
                doi = query_info["identifiers"]["doi"]
                if query_info["query_type"] == "published":
                    result = await find_published_version("biorxiv", doi)
                else:
                    result = await get_preprint_by_doi("biorxiv", doi)
            else:
                result = await get_recent_preprints("biorxiv", 7, max_results)
            return {"source": "biorxiv", "data": result}
            
    except Exception as e:
        return {"source": database, "error": str(e)}

@app.post("/deepresearch")
async def process_query(query: str = Body(..., embed=True), max_results: Optional[int] = 10):
    try:
        # Classify the query using LLM
        query_info = await classify_query(query)
        query_info["query"] = query  # Add original query
        
        # Process query for all selected databases
        results = []
        for database in query_info["databases"]:
            result = await process_database_query(database, query_info, max_results)
            results.append(result)
        
        return results
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# PubMed endpoints
@app.get("/pubmed/search")
async def pubmed_search(query: str, max_results: Optional[int] = 10):
    try:
        result = await search_pubmed(query, max_results)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pubmed/abstract/{pmid}")
async def pubmed_abstract(pmid: str):
    try:
        result = await get_pubmed_abstract(pmid)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ClinicalTrials.gov endpoints
@app.get("/clinicaltrials/search")
async def clinicaltrials_search(query: str, max_results: Optional[int] = 10):
    try:
        result = search_trials(query, max_results)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clinicaltrials/trial/{nct_id}")
async def clinicaltrials_trial(nct_id: str):
    try:
        result = get_trial_details(nct_id)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# BioRxiv endpoints
@app.get("/biorxiv/preprint/{doi}")
async def biorxiv_preprint(doi: str, server: str = "biorxiv"):
    try:
        result = await get_preprint_by_doi(server, doi)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/biorxiv/search/recent")
async def biorxiv_search_recent(server: str = "biorxiv", days: int = 7, max_results: Optional[int] = 10, category: Optional[str] = None):
    try:
        result = await get_recent_preprints(server, days, max_results, category)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True) 