import streamlit as st
import sys
import os
from dotenv import load_dotenv
import yt_dlp
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import BlockedPromptException
import textwrap
import faiss
import numpy as np
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
import requests
import re
import time

# .env 파일 로드
load_dotenv()

# MongoDB Atlas 연결
mongodb_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongodb_uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
db = client['youtube_transcripts']
videos_collection = db['videos']

# OpenAI 클라이언트 초기화
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Gemini API 키 설정
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# YouTube Data API 키 설정
youtube_api_key = os.getenv("YOUTUBE_API_KEY")

# FAISS 인덱스 초기화
dimension = 1536  # OpenAI ada-002 모델의 임베딩 차원
index = faiss.IndexFlatL2(dimension)

def initialize_faiss_index():
    all_videos = videos_collection.find({})
    for video in all_videos:
        if 'embedding' in video:
            index.add(np.array([video['embedding']], dtype=np.float32))

initialize_faiss_index()

# 헬퍼 함수들
def get_video_id(url):
    if "youtu.be" in url:
        return url.split("/")[-1]
    elif "youtube.com" in url:
        return url.split("v=")[1].split("&")[0]
    else:
        raise ValueError("올바른 YouTube URL이 아닙니다.")

def get_video_info(video_id):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails&id={video_id}&key={youtube_api_key}"
    response = requests.get(url)
    data = response.json()
    if "items" in data and len(data["items"]) > 0:
        title = data["items"][0]["snippet"]["title"]
        channel = data["items"][0]["snippet"]["channelTitle"]
        duration = data["items"][0]["contentDetails"]["duration"]
        duration_seconds = parse_duration(duration)
        return title, channel, duration_seconds
    else:
        raise ValueError("YouTube API로부터 비디오 정보를 가져올 수 없습니다.")

def parse_duration(duration):
    match = re.match(r'PT(\d+H)?(\d+M)?(\d+S)?', duration)
    hours = int(match.group(1)[:-1]) if match.group(1) else 0
    minutes = int(match.group(2)[:-1]) if match.group(2) else 0
    seconds = int(match.group(3)[:-1]) if match.group(3) else 0
    return hours * 3600 + minutes * 60 + seconds

def format_time(seconds):
    minutes, secs = divmod(int(seconds), 60)
    if minutes > 0:
        return f"{minutes}분 {secs}초"
    else:
        return f"{secs}초"

def display_question_answer(question, answer):
    st.markdown(
        f"""
        <div style="
            background-color: var(--background-color);
            color: var(--text-color);
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        ">
            <strong>질문:</strong> {question}
        </div>
        <div style="
            background-color: var(--background-color);
            color: var(--text-color);
            padding: 1rem;
            border-radius: 0.5rem;
        ">
            <strong>답변:</strong> {answer}
        </div>
        """,
        unsafe_allow_html=True
    )

def update_progress(progress_bar, status_text, estimated_time):
    start_time = time.time()
    for i in range(99):
        time.sleep(estimated_time / 100)
        progress_bar.progress(i + 1)
        status_text.text(f"변환 중... 예상 진행률: {i + 1}%")
    
    while time.time() - start_time < estimated_time + 10:
        time.sleep(0.1)
    
    status_text.text("거의 완료되었습니다. 조금만 기다려주세요 🙏")

def download_audio(url, output_path):
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

def transcribe_audio(file_path, estimated_time, progress_bar, status_text):
    update_progress(progress_bar, status_text, estimated_time)
    
    with open(file_path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    
    status_text.text(f"변환 완료: {len(transcript.text)} 글자 처리됨")
    return transcript.text

def embed_text(text):
    response = openai_client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding

def add_to_faiss_index(embedding):
    global index
    index.add(np.array([embedding], dtype=np.float32))
    return index.ntotal - 1

def initialize_session_state():
    if 'processed_videos' not in st.session_state:
        st.session_state.processed_videos = []

def add_processed_video(video_data):
    if 'processed_videos' not in st.session_state:
        st.session_state.processed_videos = []
    # 중복 방지를 위해 video_id 확인
    if not any(v['video_id'] == video_data['video_id'] for v in st.session_state.processed_videos):
        st.session_state.processed_videos.append(video_data)

def get_video_info_from_db(video_id):
    # 데이터베이스에서 비디오 정보 조회
    return videos_collection.find_one({"video_id": video_id})

def process_video(video_url, title, channel, duration):
    try:
        video_id = get_video_id(video_url)
        start_time = time.time()
        
        # 데이터베이스에서 기존 비디오 확인
        existing_video = get_video_info_from_db(video_id)
        
        if existing_video and 'transcript' in existing_video:
            st.success("이미 처리된 동영상입니다. 기존 데이터를 활용합니다.")
            end_time = time.time()
            processing_time = end_time - start_time
            st.info(f"데이터 불러오기 시간: {processing_time:.2f}초")
            
            # 세션 상태에 추가
            video_data = {
                "video_id": video_id,
                "title": existing_video['title'],
                "channel": existing_video['channel'],
                "duration": existing_video['duration'],
                "transcript_length": len(existing_video['transcript'])
            }
            add_processed_video(video_data)
            
            return existing_video['_id']
        
        # 새로운 비디오 처리
        with st.spinner("오디오 다운로드 중..."):
            output_file = f"audio_output_{video_id}"
            audio_file = download_audio(video_url, output_file)
        
        with st.spinner("오디오 텍스트 변환 중..."):
            estimated_time = duration * 0.3
            formatted_time = format_time(estimated_time)
            st.info(f"예상 소요 시간: 약 {estimated_time:.2f}초 (10분 길이 영상 기준 약 2분 소요)")

            progress_bar = st.progress(0)
            status_text = st.empty()

            transcript = transcribe_audio(audio_file, estimated_time, progress_bar, status_text)

        transcript_length = len(transcript)
        
        with st.spinner("임베딩 생성 중..."):
            embedding = embed_text(transcript)
        
        with st.spinner("FAISS 인덱스에 추가 중..."):
            faiss_index = add_to_faiss_index(embedding)
        
        with st.spinner("MongoDB에 저장 중..."):
            video_data = {
                "video_id": video_id,
                "title": title,
                "channel": channel,
                "duration": duration,
                "transcript": transcript,
                "transcript_length": transcript_length,
                "embedding": embedding
            }
            result = videos_collection.insert_one(video_data)
        
        os.remove(audio_file)
        
        end_time = time.time()
        processing_time = end_time - start_time
        st.success(f"비디오 처리 성공! ID: {result.inserted_id}")
        st.info(f"총 처리 시간: {processing_time:.2f}초")
        
        # 세션 상태에 추가
        session_video_data = {
            "video_id": video_id,
            "title": title,
            "channel": channel,
            "duration": duration,
            "transcript_length": transcript_length
        }
        add_processed_video(session_video_data)
        
        return result.inserted_id
    except Exception as e:
        st.error(f"처리 중 예상치 못한 오류 발생: {str(e)}")
    return None

def search_similar_transcripts(query, top_k=3):
    query_vector = embed_text(query)
    D, I = index.search(np.array([query_vector], dtype=np.float32), top_k)
    return [int(i) for i in I[0]]

def generate_response(query, relevant_transcripts):
    model = genai.GenerativeModel('gemini-pro')
    
    combined_transcript = "\n\n".join(relevant_transcripts)
    
    prompt = textwrap.dedent(f"""
    다음은 YouTube 비디오의 전체 내용입니다:

    {combined_transcript}

    질문: {query}

    위의 내용을 바탕으로 질문에 답변해주세요. 답변 시 다음 지침을 따라주세요:
    1. 주어진 내용에서 직접적으로 관련된 정보를 찾아 답변하세요.
    2. 정보가 부족하거나 관련이 없는 경우, "제공된 내용에는 이 질문에 답할 만한 충분한 정보가 없습니다."라고 명시하세요.
    3. 추측하지 말고, 주어진 내용에 실제로 있는 정보만 사용하세요.
    4. 가능하다면 관련 부분을 간접적으로 인용하여 답변의 근거를 제시하세요.
    5. 의학적 조언이나 전문적인 내용을 다룰 때는 "영상에서 언급된 바에 따르면"이라는 문구로 시작하세요.

    답변:
    """)

    try:
        response = model.generate_content(prompt, safety_settings=[
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ])
        
        if response.parts:
            return response.text
        else:
            return "답변을 생성할 수 없습니다. 다른 질문을 시도해 보세요."
    except Exception as e:
        error_message = str(e)
        if "SafetyError" in error_message:
            return "안전 정책으로 인해 답변 생성이 중단되었습니다. 다른 질문을 시도해 보세요."
        elif "create() got an unexpected keyword argument" in error_message:
            return "내부 API 호출에 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
        else:
            return f"처리 중 예상치 못한 오류가 발생했습니다: {error_message}"

def view_full_transcript(video_id):
    video = get_video_info_from_db(video_id)
    if video and 'transcript' in video:
        st.markdown(f"### {video['title']}")
        st.caption(f"채널: {video['channel']}")
        st.markdown("---")

        st.markdown("""
        <style>
        .transcript-box {
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            max-height: 500px;
            overflow-y: auto;
        }
        .light-mode {
            background-color: white;
            color: black;
        }
        .dark-mode {
            background-color: #262730;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

        is_dark_mode = st.get_option("theme.base") == "dark"
        theme_class = "dark-mode" if is_dark_mode else "light-mode"

        st.markdown(f"<div class='transcript-box {theme_class}'>{video['transcript']}</div>", unsafe_allow_html=True)
    else:
        st.error("텍스트 변환 내용 데이터를 찾을 수 없습니다. 해당 비디오의 데이터가 삭제되었거나 처리 과정에 문제가 있었을 수 있습니다.")
        st.info("처리된 비디오 목록으로 돌아가려면 사이드바의 '처리된 비디오 보기' 버튼을 클릭하세요.")

def sidebar_navigation():
    st.sidebar.title("네비게이션")
    if st.sidebar.button("새 비디오 처리"):
        st.session_state.page = "새 비디오 처리"
    if st.sidebar.button("질문하기"):
        st.session_state.page = "질문하기"
    if st.sidebar.button("처리된 비디오 보기"):
        st.session_state.page = "처리된 비디오 보기"

def new_video_processing():
    st.header("새 YouTube 비디오 처리")
    st.warning("주의: 현재 20분 이하의 영상만 처리 가능합니다.")
    
    video_url = st.text_input("YouTube 비디오 URL 입력")
    if st.button("비디오 처리"):
        try:
            video_id = get_video_id(video_url)
            with st.spinner("비디오 정보 가져오는 중..."):
                title, channel, duration = get_video_info(video_id)
            
            if duration > 1200:  # 20분 = 1200초
                st.error(f"죄송합니다. 이 비디오는 {duration//60}분 {duration%60}초로, 20분을 초과하여 처리할 수 없습니다.")
            else:
                result = process_video(video_url, title, channel, duration)
                if result:
                    st.session_state.show_buttons = True  # 버튼 표시 상태를 세션에 저장
                    st.session_state.page = "비디오 처리 완료"
                    st.rerun()
        except ValueError as ve:
            st.error(f"입력 오류: {str(ve)}")
        except Exception as e:
            st.error(f"처리 중 예상치 못한 오류 발생: {str(e)}")
    
    if st.session_state.get('page') == "비디오 처리 완료":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("질문하기", key="ask_question_button"):
                st.session_state.page = "질문하기"
                st.rerun()
        with col2:
            if st.button("처리된 비디오 목록보기", key="view_videos_button"):
                st.session_state.page = "처리된 비디오 보기"
                st.rerun()

def view_processed_videos():
    st.header("처리된 비디오 목록")
    if not st.session_state.processed_videos:
        st.info("이 세션에서 처리된 비디오가 없습니다. 새 비디오를 처리해 주세요.")
    else:
        for index, video in enumerate(st.session_state.processed_videos):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"제목: {video['title']}")
                st.write(f"채널명: {video['channel']}")
                duration = video['duration']
                st.write(f"길이: {duration // 60}분 {duration % 60}초")
                st.write(f"텍스트 길이: {video['transcript_length']}글자")
            with col2:
                if st.button("채팅하기", key=f"chat_{video['video_id']}_{index}"):
                    st.session_state.page = "질문하기"
                    st.session_state.selected_video_id = video['video_id']
                    st.rerun()
            with col3:
                if st.button("전문 보기", key=f"full_{video['video_id']}_{index}"):
                    st.session_state.page = "전체 스크립트 보기"
                    st.session_state.selected_video_id = video['video_id']
                    st.rerun()
            st.write("---")

def ask_question():
    st.header("처리된 비디오에 대해 질문하기")
    if not st.session_state.processed_videos:
        st.info("이 세션에서 처리된 비디오가 없습니다. 먼저 비디오를 처리해주세요.")
    else:
        video_list = [(v['video_id'], f"{v['title']} - {v['channel']}") for v in st.session_state.processed_videos]
        
        default_index = 0
        if 'selected_video_id' in st.session_state:
            default_index = next((i for i, v in enumerate(video_list) if v[0] == st.session_state.selected_video_id), 0)
        
        selected_video = st.selectbox("비디오 선택", video_list, index=default_index, format_func=lambda x: x[1])
        st.session_state.selected_video_id = selected_video[0]

        query = st.text_input("질문을 입력하세요", placeholder="영상을 10줄 정도로 요약해주세요")
        if st.button("답변 받기"):
            with st.spinner("답변 생성 중..."):
                try:
                    video_data = get_video_info_from_db(st.session_state.selected_video_id)
                    if video_data and 'transcript' in video_data:
                        response = generate_response(query, [video_data['transcript']])
                        
                        st.markdown(f"**입력한 질문:** {query}")
                        st.markdown("---")
                        display_question_answer(query, response)
                    else:
                        st.error("선택한 비디오의 트랜스크립트를 찾을 수 없습니다.")
                except Exception as e:
                    st.error(f"답변 생성 중 오류가 발생했습니다: {str(e)}")

def main():
    st.set_page_config(page_title="YouTube 비디오 Q&A", page_icon="🎥")
    st.title("YouTube 비디오 Q&A")

    initialize_session_state()

    if 'page' not in st.session_state:
        st.session_state.page = "새 비디오 처리"

    sidebar_navigation()

    if st.session_state.page == "새 비디오 처리":
        new_video_processing()
    elif st.session_state.page == "질문하기":
        ask_question()
    elif st.session_state.page == "처리된 비디오 보기":
        view_processed_videos()
    elif st.session_state.page == "전체 스크립트 보기":
        view_full_transcript(st.session_state.selected_video_id)
        if st.button("처리된 비디오 목록으로 돌아가기"):
            st.session_state.page = "처리된 비디오 보기"
            st.rerun()

if __name__ == "__main__":
    main()
