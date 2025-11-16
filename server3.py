"""
FastAPI server for ticket classification with improved error handling.
"""

import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
from datetime import datetime
import traceback

from classify3 import classify
from db_manager import ChromaDBManager
from models import BatchClassificationRequest
from logging_config import setup_logging, get_logger
from config import config

# Setup logging
setup_logging(config.log_level)
logger = get_logger(__name__)

app = FastAPI(
    title="LLM Ticket Classification API",
    description="API for classifying customer support tickets using LLM",
    version="1.0.0"
)

# Initialize DB manager
db_manager = ChromaDBManager()


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status.
    """
    try:
        # Check ChromaDB connection
        is_healthy = db_manager.health_check()
        
        if not is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ChromaDB connection failed"
            )
        
        return {
            "status": "healthy",
            "chromadb": "connected",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/classify/")
async def classify_logs(request: Request, file: Optional[UploadFile] = None):
    """
    Classify tickets from either a CSV file upload or JSON body.
    
    Accepts:
    - CSV file with 'channel' and 'message_content' columns
    - JSON body with 'message_content' (list) and optional 'channel'
    
    Returns:
    - CSV file with classification results
    """
    file_handle = None
    try:
        # If a file is uploaded
        if file:
            if not file.filename or not file.filename.endswith('.csv'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File must be a CSV file."
                )

            try:
                file_handle = file.file
                df = pd.read_csv(file_handle, encoding='ISO-8859-1')
            except pd.errors.EmptyDataError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="CSV file is empty."
                )
            except Exception as e:
                logger.error(f"Error reading CSV: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Error reading CSV file: {str(e)}"
                )
            
            required_columns = ["channel", "message_content"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"CSV must contain columns: {required_columns}. Missing: {missing_columns}"
                )

            logs = list(zip(df["channel"], df["message_content"]))
            logger.info(f"Received CSV file with {len(logs)} entries")

        # If JSON body is provided instead of a file
        else:
            try:
                body = await request.json()
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON in request body: {str(e)}"
                )
            
            # Validate using Pydantic model
            try:
                request_data = BatchClassificationRequest(**body)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid request format: {str(e)}"
                )
            
            channel = request_data.channel or "unknown"
            logs = [(channel, msg) for msg in request_data.messages]
            df = pd.DataFrame(logs, columns=["channel", "message_content"])
            logger.info(f"Received JSON request with {len(logs)} messages")

        # Validate logs
        if not logs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No logs to classify."
            )

        # Classify the logs
        try:
            labels, routing_info, processing_costs, chroma_ids = classify(logs)
        except Exception as e:
            logger.error(f"Classification error: {e}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Classification failed. Please try again later."
            )

        # Append results
        df["target_label"] = labels
        df["routing_info"] = routing_info
        df["processing_cost"] = processing_costs
        df["chroma_vector_id"] = chroma_ids

        # Save to CSV and return file
        output_file = "output2.csv"
        try:
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error saving results file."
            )
        
        return FileResponse(
            output_file,
            media_type='text/csv',
            filename="classification_results.csv"
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Required file not found"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred. Please try again later."
        )
    finally:
        if file_handle:
            try:
                file_handle.close()
            except Exception:
                pass

