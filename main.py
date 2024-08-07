from fastapi import FastAPI
from api.routers import aging as aging_router
from dotenv import load_dotenv

load_dotenv()  # This loads the variables from .env

app = FastAPI()
app.include_router(aging_router.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)