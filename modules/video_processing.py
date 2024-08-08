import os
import logging
from datetime import datetime
import re
from urllib.parse import urlparse, parse_qs
import requests
import isodate
import yt_dlp
import time
from config import MAX_VIDEO_DURATION, YOUTUBE_API_KEY
from modules.database import videos_collection
from modules.nlp import transcribe_audio, embed_text
from openai import OpenAI
import tiktoken
from config import OPENAI_API_KEY

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chunk_text(text, max_tokens=8000):
    """텍스트를 지정된 최대 토큰 수로 나눕니다."""
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = enc.encode(text)
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    for token in tokens:
        if current_chunk_tokens + 1 > max_tokens:
            chunks.append(enc.decode(current_chunk))
            current_chunk = []
            current_chunk_tokens = 0
        current_chunk.append(token)
        current_chunk_tokens += 1

    if current_chunk:
        chunks.append(enc.decode(current_chunk))

    return chunks

def embed_text(text):
    """텍스트를 청크로 나누고 각 청크를 임베딩합니다."""
    chunks = chunk_text(text)
    embeddings = []

    for chunk in chunks:
        response = client.embeddings.create(input=[chunk], model="text-embedding-ada-002")
        embeddings.append(response.data[0].embedding)

    # 모든 청크의 임베딩 평균을 계산
    if embeddings:
        avg_embedding = [sum(x) / len(embeddings) for x in zip(*embeddings)]
        return avg_embedding
    else:
        return []

def transcribe_audio(file_path):
    """오디오 파일을 텍스트로 변환"""
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

def extract_video_id_and_process(url):
    """
    YouTube URL에서 비디오 ID를 추출하고 적절한 형식으로 처리합니다.

    :param url: YouTube 비디오 URL (기본 또는 단축 형식)
    :return: 기본 YouTube URL 형식과 비디오 ID
    """
    logger.info(f"처리 중인 URL: {url}")

    try:
        parsed_url = urlparse(url)
        logger.debug(f"파싱된 URL: {parsed_url}")

        # youtu.be 형식 처리
        if 'youtu.be' in parsed_url.netloc:
            video_id = parsed_url.path.lstrip('/')
            logger.info(f"youtu.be 형식에서 추출된 비디오 ID: {video_id}")
        # youtube.com 형식 처리
        elif 'youtube.com' in parsed_url.netloc:
            if 'v' in parse_qs(parsed_url.query):
                video_id = parse_qs(parsed_url.query)['v'][0]
                logger.info(f"youtube.com 쿼리 파라미터에서 추출된 비디오 ID: {video_id}")
            elif 'embed' in parsed_url.path:
                video_id = parsed_url.path.split('/')[-1]
                logger.info(f"youtube.com 임베드 URL에서 추출된 비디오 ID: {video_id}")
            elif 'shorts' in parsed_url.path:
                video_id = parsed_url.path.split('/')[-1]
                logger.info(f"youtube.com 쇼츠 URL에서 추출된 비디오 ID: {video_id}")
            else:
                # 기타 youtube.com URL 형식 처리
                match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
                if match:
                    video_id = match.group(1)
                    logger.info(f"정규식을 통해 추출된 비디오 ID: {video_id}")
                else:
                    raise ValueError("YouTube 비디오 ID를 찾을 수 없습니다.")
        else:
            # 기타 형식 (예: 내장 URL)
            match = re.search(r'(?:embed\/|v\/|vi\/|e\/|shorts\/|watch\?v=)([^#\&\?]{11})', url)
            if match:
                video_id = match.group(1)
                logger.info(f"기타 URL 형식에서 추출된 비디오 ID: {video_id}")
            else:
                raise ValueError("지원되지 않는 YouTube URL 형식입니다.")

        if not video_id:
            raise ValueError("YouTube 비디오 ID를 추출할 수 없습니다.")

        # 기본 URL 형식으로 변환
        base_url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info(f"생성된 기본 URL: {base_url}")

        return base_url, video_id

    except Exception as e:
        logger.error(f"URL 파싱 중 오류 발생: {str(e)}")
        raise ValueError(f"YouTube URL 처리 중 오류 발생: {str(e)}")


def get_video_info(video_url):
    """YouTube API를 사용하여 비디오 정보를 가져옵니다."""
    try:
        base_url, video_id = extract_video_id_and_process(video_url)
        logger.info(f"API 요청을 위한 비디오 ID: {video_id}")

        url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails&id={video_id}&key={YOUTUBE_API_KEY}"
        logger.debug(f"YouTube API 요청 URL: {url}")

        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "items" in data and len(data["items"]) > 0:
            video_data = data["items"][0]
            title = video_data["snippet"]["title"]
            channel = video_data["snippet"]["channelTitle"]
            duration = parse_duration(video_data["contentDetails"]["duration"])
            logger.info(f"비디오 정보 추출 성공 - 제목: {title}, 채널: {channel}, 길이: {duration}초")
            return title, channel, duration
        else:
            raise ValueError(f"비디오를 찾을 수 없습니다. 비디오 ID: {video_id}")

    except requests.exceptions.RequestException as e:
        logger.error(f"YouTube API 요청 중 오류 발생: {e}")
        raise ValueError(f"비디오 정보를 가져오는 중 오류가 발생했습니다. (Video ID: {video_id})")
    except ValueError as e:
        logger.error(f"비디오 정보 처리 중 오류 발생: {e}")
        raise
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {e}")
        raise ValueError(f"비디오 정보를 처리하는 중 오류가 발생했습니다: {str(e)}")

def parse_duration(duration):
    """YouTube API의 duration 문자열을 초 단위로 변환합니다."""
    return int(isodate.parse_duration(duration).total_seconds())


def get_video_captions(video_id):
    """YouTube API를 사용하여 비디오의 자막을 가져옵니다."""
    url = f"https://www.googleapis.com/youtube/v3/captions?part=snippet&videoId={video_id}&key={YOUTUBE_API_KEY}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "items" in data and len(data["items"]) > 0:
            ko_caption = next((item for item in data["items"] if item["snippet"]["language"] == "ko"), None)
            en_caption = next((item for item in data["items"] if item["snippet"]["language"] == "en"), None)

            caption_id = ko_caption["id"] if ko_caption else (
                en_caption["id"] if en_caption else data["items"][0]["id"])

            return download_caption(caption_id)
        else:
            logger.info(f"비디오 {video_id}에 사용 가능한 자막이 없습니다.")
            return None

    except requests.RequestException as e:
        logger.error(f"자막 정보 요청 중 오류 발생: {str(e)}")
        return None


def download_caption(caption_id):
    """지정된 자막 ID의 자막 내용을 다운로드합니다."""
    url = f"https://www.googleapis.com/youtube/v3/captions/{caption_id}?key={YOUTUBE_API_KEY}"

    try:
        response = requests.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        caption_data = response.json()

        return caption_data.get("text", "")

    except requests.RequestException as e:
        logger.error(f"자막 다운로드 중 오류 발생: {str(e)}")
        return None


def process_video(video_url, user_id, progress_bar=None):
    try:
        # URL인지 비디오 ID인지 확인
        if 'youtube.com' in video_url or 'youtu.be' in video_url:
            # URL 정규화 및 비디오 ID 추출
            normalized_url, video_id = extract_video_id_and_process(video_url)
        else:
            # 입력이 이미 비디오 ID인 경우
            video_id = video_url
            normalized_url = f"https://www.youtube.com/watch?v={video_id}"

        logger.info(f"처리할 비디오 ID: {video_id}")

        # 기존 처리된 비디오 확인
        existing_video = get_existing_video(video_id)
        if existing_video:
            logger.info(f"비디오 ID {video_id}는 이미 처리되었습니다. 기존 데이터를 사용합니다.")
            update_user_for_video(existing_video['_id'], user_id)
            return existing_video['_id']

        # 새 비디오 처리 로직
        title, channel, duration = get_video_info(normalized_url)

        if duration > MAX_VIDEO_DURATION:
            raise ValueError(f"비디오 길이가 {MAX_VIDEO_DURATION // 60}분을 초과합니다.")

        # 자막 데이터 가져오기 시도
        caption_text = get_video_captions(video_id)
        if progress_bar:
            if caption_text:
                progress_bar.progress(20, text="자막 다운로드 성공! 🥳")
            else:
                progress_bar.progress(20, text="자막 다운로드 실패 😔 오디오 변환 시도 중...")

        if caption_text:
            logger.info("자막 데이터를 성공적으로 가져왔습니다.")
            transcript = caption_text
        else:
            logger.info("자막을 가져올 수 없어 오디오 변환을 시도합니다.")
            if progress_bar:
                progress_bar.progress(30, text="영상 다운로드 중... 🌎")
            audio_file = download_and_process_audio(normalized_url, video_id)
            if progress_bar:
                progress_bar.progress(45, text="영상을 텍스트로 변환 중... 💬")
            transcript = transcribe_audio(audio_file)
            os.remove(audio_file)

        if progress_bar:
            progress_bar.progress(90, text="텍스트 임베딩 중... 🤖")
        embedding = embed_text(transcript)

        video_data = {
            "video_id": video_id,
            "user_ids": [user_id],
            "title": title,
            "channel": channel,
            "duration": duration,
            "transcript": transcript,
            "embedding": embedding,
            "source": "caption" if caption_text else "audio_transcription",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "processed_at": datetime.utcnow(),
            "transcript_length": len(transcript),
            "tags": []  # 새로운 필드: 태그 (빈 리스트로 초기화)
        }

        if progress_bar:
            progress_bar.progress(100, text="DB 저장 완료! ✅")  # 진행률 100%로 설정
        result = videos_collection.insert_one(video_data)
        return result.inserted_id

    except Exception as e:
        logger.error(f"비디오 처리 중 오류 발생: {str(e)}")
        raise

def download_and_process_audio(url, video_id):
    output_path = f"temp_audio_{video_id}"
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [],
            'outtmpl': output_path + '.%(ext)s',
            'keepvideo': False,
            'noplaylist': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

        return filename
    except Exception as e:
        logger.error(f"오디오 다운로드 중 오류 발생: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise


def update_user_for_video(video_id, user_id):
    videos_collection.update_one(
        {"_id": video_id},
        {"$addToSet": {"user_ids": user_id}}
    )


def get_existing_video(video_id):
    """데이터베이스에서 기존 처리된 비디오를 찾습니다."""
    return videos_collection.find_one({"video_id": video_id})


def format_time(seconds):
    """초 단위의 시간을 읽기 쉬운 형식으로 변환합니다."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}시간 {minutes}분 {secs}초"
    elif minutes > 0:
        return f"{minutes}분 {secs}초"
    else:
        return f"{secs}초"