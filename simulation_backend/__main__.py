"""
python -m simulation_backend
"""
import uvicorn
from simulation_backend.core.settings import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run(
        "simulation_backend.app:app",
        host=s.host,
        port=s.port,
        reload=s.reload,
    )
