# database.py

from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
from config import MONGODB_URI
from datetime import datetime

# MongoDB 연결 설정
client = MongoClient(MONGODB_URI, server_api=ServerApi('1'), tlsCAFile=certifi.where())
db = client['youtube_transcripts']
users_collection = db['users']
videos_collection = db['videos']

def get_video_info_from_db(video_ids):
    """데이터베이스에서 여러 비디오 정보 조회"""
    return list(videos_collection.find({"video_id": {"$in": video_ids}}))


def get_user_videos(user_id, selected_tags=None, start_date=None, end_date=None, show_no_tags=False):
    """사용자의 처리된 비디오 목록 가져오기 (필터링 포함)"""
    query = {"user_ids": user_id}

    if show_no_tags:
        query["$or"] = [{"tags": {"$exists": False}}, {"tags": []}]
    elif selected_tags:
        query["tags"] = {"$in": selected_tags}

    if start_date and end_date:
        query["processed_at"] = {
            "$gte": start_date,
            "$lte": end_date
        }

    return list(videos_collection.find(query))


def remove_tag_from_video(video_id, tag):
    """비디오에서 태그 제거"""
    try:
        result = videos_collection.update_one(
            {"video_id": video_id},
            {"$pull": {"tags": tag}}
        )
        if result.modified_count > 0:
            logger.info(f"태그 '{tag}'가 비디오 ID {video_id}에서 성공적으로 제거되었습니다.")
            return True
        else:
            logger.warning(f"태그 '{tag}'를 비디오 ID {video_id}에서 제거하지 못했습니다. 태그가 존재하지 않을 수 있습니다.")
            return False
    except Exception as e:
        logger.error(f"태그 제거 중 오류 발생: {str(e)}")
        return False

def save_feedback(user_id, feedback):
    """피드백을 데이터베이스에 저장합니다."""
    feedback_data = {
        "user_id": user_id,
        "feedback": feedback,
        "timestamp": datetime.utcnow()
    }
    db['feedback'].insert_one(feedback_data)


def add_tag_to_video(video_id, tag):
    """비디오에 태그 추가 (최대 3개)"""
    video = videos_collection.find_one({"video_id": video_id})
    if video and len(video.get("tags", [])) < 3:
        videos_collection.update_one(
            {"video_id": video_id},
            {"$addToSet": {"tags": tag}}
        )
        return True
    return False


def remove_tag_from_video(video_id, tag):
    """비디오에서 태그 제거"""
    videos_collection.update_one(
        {"video_id": video_id},
        {"$pull": {"tags": tag}}
    )


def get_all_tags():
    """모든 고유 태그 가져오기"""
    all_tags = videos_collection.distinct("tags")
    return [tag for tag in all_tags if tag is not None]  # None 값 제거

def get_videos_by_tags(tags):
    """태그 리스트에 해당하는 비디오 정보 가져오기"""
    return list(videos_collection.find({"tags": {"$in": tags}}))