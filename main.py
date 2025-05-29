import os
import uvicorn
from plant_disease_api import app

if __name__ == "__main__":
    # Get port from environment variable or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    # Configure uvicorn server
    uvicorn.run(
        "plant_disease_api:app",
        host="0.0.0.0",
        port=port,
        workers=2,
        timeout_keep_alive=75,
        log_level="info",
        reload=False  # Disable reload in production
    )
