import httpx

API_URL = "https://example.com/ai-api"  # AI API 엔드포인트

async def fetch_ai_description(object_id: str):
    """외부 API 호출 (비동기)"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}?id={object_id}")
        if response.status_code == 200:
            return response.json().get("description")
    return None