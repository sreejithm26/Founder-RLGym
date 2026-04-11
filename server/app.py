import uvicorn
import os

def main() -> None:
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("founder_gym.api.server:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
