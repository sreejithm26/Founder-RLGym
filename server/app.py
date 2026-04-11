import uvicorn
import os

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("founder_gym.api.server:app", host="0.0.0.0", port=port, reload=True)
