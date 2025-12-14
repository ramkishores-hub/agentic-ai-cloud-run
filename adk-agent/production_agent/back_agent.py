import os
os.environ["GOOGLE_API_KEY"]=''
os.environ["GOOGLE_GENAI_USE_VERTEXAI"]='FALSE'
from pathlib import Path
import logging
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
import google.auth
from google.adk.tools.bigquery import BigQueryToolset, BigQueryCredentialsConfig
from google.adk.tools.bigquery.config import BigQueryToolConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.adk.runners import InMemoryRunner
from google.adk.tools.bigquery.config import WriteMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title = "Backend ETL Agent")
# 1. Authenticate (Uses Cloud Run Service Account automatically)
try:
   credentials, project = google.auth.default()
   tool_config = BigQueryToolConfig(write_mode=WriteMode.ALLOWED)
   bq_config = BigQueryCredentialsConfig(credentials=credentials)
   os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project)

   # 2. Initialize the Toolset
   # We give the agents permissions to list tables and RUN queries.

   bq_tools = BigQueryToolset(
    credentials_config=bq_config,
    tool_filter=['list_dataset_ids', 'get_table_info', 'execute_sql'],
    bigquery_tool_config=tool_config
   )
except Exception as e:
    print('Error in Service Account',e)
# Load environment variables
root_dir = Path(__file__).parent.parent
dotenv_path = root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "europe-west1")

# # Configure model connection
# gemma_model_name = os.getenv("GEMMA_MODEL_NAME", "gemma3:270m")
# api_base = os.getenv("OLLAMA_API_BASE", "localhost:10010")  # Location of Ollama server

# Back end Data Engineering Agent - GPU-accelerated conversational assistant
back_end_agent = Agent(
   model="gemini-2.0-flash",
   name="data_engineering_agent",
   description="Data Engineering agent which checks data quality in Medallion Architecture.",
   instruction="""
   You are an autonomous Data Engineer managing a Medallion Architecture.Your sole tool for all data operations (INSERT, SELECT, UPDATE, MERGE, etc.) is the **`execute_sql`** function within the `bq_tools` toolset. 
    
    YOUR OBJECTIVE is to Construct the SQL statements and execute the statements to complete the following steps:
    1. SCAN: Look at `ai-agent-with-cloud-run.medallion_arch.raw_orders` where processing_status is NULL and all columns are STRING.
    2. VALIDATE & TRANSFORM (Bronze -> Silver):
        INSERT into `ai-agent-with-cloud-run.medallion_arch.clean_orders` FROM from `ai-agent-with-cloud-run.medallion_arch.raw_orders` considering the below scenarios.
        Consider the following mapping while inserting the data into `ai-agent-with-cloud-run.medallion_arch.clean_orders`.
        `ai-agent-with-cloud-run.medallion_arch.raw_orders.order_id` -> `ai-agent-with-cloud-run.medallion_arch.clean_orders.order_id`
        `ai-agent-with-cloud-run.medallion_arch.raw_orders.customer_email` -> `ai-agent-with-cloud-run.medallion_arch.clean_orders.customer_email`
        `ai-agent-with-cloud-run.medallion_arch.raw_orders.amount` -> `ai-agent-with-cloud-run.medallion_arch.clean_orders.amount`
        `ai-agent-with-cloud-run.medallion_arch.raw_orders.order_date` -> `ai-agent-with-cloud-run.medallion_arch.clean_orders.order_date`
       - If order_id contains garbage values, do NOT insert the rows into Silver.
       - Select rows where amount is positive and customer_email contains '@'.
       - If order_date is in a weird format, standardize it to YYYY-MM-DD.
       - Insert these valid rows into `ai-agent-with-cloud-run.medallion_arch.clean_orders`.
       - If SAFE_CAST returns NULL (meaning data is invalid), do NOT insert into `ai-agent-with-cloud-run.medallion_arch.clean_orders`.
       - CAST amount to FLOAT64, CAST order_date to DATE while maintaining the data type for customer_email and order_id.
    3. AGGREGATE (Silver -> Gold):
        INSERT data INTO `ai-agent-with-cloud-run.medallion_arch.daily_sales_summary` from `ai-agent-with-cloud-run.medallion_arch.clean_orders` considering the below scenarios.
        total_amount should be calculated using amount by doing aggregation on customer_email.
        total_orders should be calculated using order_id by doing aggregation on customer_email.
       - Calculate total_amount and total_orders aggregated on customer_email using `ai-agent-with-cloud-run.medallion_arch.clean_orders`.
       - MERGE/UPDATE/INSERT the results into `ai-agent-with-cloud-run.medallion_arch.daily_sales_summary`.
    4. CLEANUP:
       - Update the `ai-agent-with-cloud-run.medallion_arch.raw_orders` rows by updating the column processing_status = 'PROCESSED' after the above steps are completed.
    
    OUTPUT: Return a JSON summary of how many rows were processed for each step.
    """,
   tools=[bq_tools]
)

runner = InMemoryRunner(agent=back_end_agent)

class ETLResponse(BaseModel):
    status: str
    agent_response: str


# --- API ENDPOINTS ---

@app.get("/")
def health_check():
    """Simple health check for Cloud Run."""
    return {"status": "ok", "service": "Backend Agent"}

@app.post("/run-etl", response_model=ETLResponse)
async def run_etl_endpoint():
    """
    Triggered by Cloud Scheduler.
    """
    try:
        system_trigger = "Check for unprocessed data in Bronze and run the full pipeline."
        
        # Check if tools loaded correctly
        if not bq_tools:
            raise HTTPException(status_code=500, detail="BigQuery Tools failed to initialize.")

        response = await runner.run_debug(system_trigger)
        final_response = ""
        if response:
            if response[0].content:
                if response[0].content.parts:
                    if response[0].content.parts[0]:
                        final_response = response[0].content.parts[0].text
        
        return {
            "status": "success", 
            "agent_response": final_response
        }
    except Exception as e:
        logger.error(f"ETL Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ETL Failed: {str(e)}")

# --- LOCAL EXECUTION ---
if __name__ == "__main__":
    import uvicorn
    # This is for testing locally with 'python back_agent.py'
    uvicorn.run(app, host="0.0.0.0", port=8080)
