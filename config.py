import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수 설정
MONGODB_URI = os.getenv("MONGODB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# 기타 설정
MAX_VIDEO_DURATION = 1200  # 20분 (초 단위)