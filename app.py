from fastapi import FastAPI
from MovieRecomend import recommend_for_user

app = FastAPI()

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int):
    result = recommend_for_user(user_id)
    return {
        "user_id": user_id,
        "recommendations": result
    }