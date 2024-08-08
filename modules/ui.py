import streamlit as st
from modules import auth, video_processing, database, nlp
import time
import logging
from datetime import datetime, timedelta  # timedelta를 추가로 import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def show_sidebar():
    with st.sidebar:
        st.write(f"환영합니다, {st.session_state.user['username']}님!")
        if st.button("로그아웃"):
            st.session_state.user = None
            st.session_state.page = 'login'
            st.rerun()

        st.write("---")
        if st.button("새 영상 처리"):
            st.session_state.page = 'process_video'
        if st.button("질문하기"):
            st.session_state.page = 'ask_question'
        if st.button("처리된 영상 목록보기"):
            st.session_state.page = 'view_videos'

def show_login_form():
    st.markdown(
        """
        <style>
        .login-container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="login-container">', unsafe_allow_html=True)

    st.subheader("로그인")

    tab1, tab2 = st.tabs(["로그인", "회원가입"])

    with tab1:
        username = st.text_input("사용자명")
        password = st.text_input("비밀번호", type="password")
        if st.button("로그인"):
            user = auth.authenticate_user(username, password)
            if user:
                st.success("로그인 성공!")
                st.session_state.user = user
                st.session_state.page = 'process_video'
                st.rerun()
            else:
                st.error("로그인 실패. 사용자명과 비밀번호를 확인하세요.")

    with tab2:
        new_username = st.text_input("새 사용자명", placeholder="한글 아이디도 사용이 가능해요")
        new_password = st.text_input("새 비밀번호", type="password", placeholder="평소 사용하지 않는 비밀번호를 사용하세요")
        if st.button("회원가입"):
            if auth.register_user(new_username, new_password):
                st.success("회원가입 성공! 이제 로그인할 수 있습니다.")
            else:
                st.error("회원가입 실패. 이미 존재하는 사용자명입니다.")

    st.markdown('</div>', unsafe_allow_html=True)

def show_video_processing_form():
    st.header("새 YouTube 영상 처리")
    st.warning(f"주의: 현재 {video_processing.MAX_VIDEO_DURATION // 60}분 이하의 영상만 처리 가능합니다.")

    video_url = st.text_input("YouTube 영상 URL 입력")
    if st.button("영상 처리", key="process_video_button"):
        if not video_url:
            st.error("YouTube 영상 URL을 입력해주세요.")
            return

        try:
            user_id = st.session_state.user['_id']
            with st.spinner("영상 정보 가져오는 중... ⏳"):
                title, channel, duration = video_processing.get_video_info(video_url)
                estimated_time = (duration // 600) * 60 + (duration % 600) // 10  # 10분당 60초 기준 계산
                st.info(f"**{title}** ({channel}) - 예상 처리 시간: 약 {estimated_time}초 ⏰")

            # 기존에 처리된 영상인지 확인
            _, video_id = video_processing.extract_video_id_and_process(video_url)
            existing_video = video_processing.get_existing_video(video_id)

            if existing_video:
                st.info(f"이 영상는 이미 처리되었습니다. 기존 데이터를 사용합니다.")
                video_processing.update_user_for_video(existing_video['_id'], user_id)
                video_id = existing_video['_id']
            else:
                progress_bar = st.progress(0, text="영상 처리 중... 🏃")
                start_time = time.time()

                video_id = video_processing.process_video(video_url, user_id, progress_bar)

                end_time = time.time()
                elapsed_time = end_time - start_time
                st.success(f"영상 처리 완료! 🎉  ({video_processing.format_time(elapsed_time)} 소요)")

            update_processed_videos(user_id)

            # 처리 완료 후 버튼 표시
            col1, col2 = st.columns(2)
            with col1:
                if st.button("질문하기", key="ask_question_button"):
                    st.session_state.next_page = "ask_question"
                    st.session_state.current_selected_video_id = video_id
            with col2:
                if st.button("영상 목록 보기", key="view_videos_button"):
                    st.session_state.next_page = "view_videos"

        except Exception as e:
            st.error(f"영상 처리 중 오류 발생: {str(e)}")


def show_question_form():
    st.header("영상에 대해 질문하기")
    user_id = st.session_state.user['_id']

    # 질문 모드 선택
    question_mode = st.radio("질문 모드 선택", ["하나의 영상 기반 질문", "태그에 포함된 다수 영상 기반 질문"])

    if question_mode == "하나의 영상 기반 질문":
        show_individual_video_question(user_id)
    else:
        show_tag_based_question(user_id)


def show_individual_video_question(user_id):
    user_videos = database.get_user_videos(user_id)
    if user_videos:
        video_options = {f"{v['title']} - {v['channel']}": v['video_id'] for v in user_videos}
        selected_video_title = st.selectbox("영상 선택", list(video_options.keys()), key="individual_video_selector")
        selected_video_id = video_options[selected_video_title]

        question = st.text_input("질문을 입력하세요")
        if st.button("답변 받기"):
            if question:
                with st.spinner("답변 생성 중..."):
                    try:
                        video_data = database.get_video_info_from_db([selected_video_id])
                        if video_data and 'transcript' in video_data[0]:
                            response = nlp.generate_response(question, [video_data[0]['transcript']])
                            display_response(question, response)
                        else:
                            st.error("선택한 영상의 트랜스크립트를 찾을 수 없습니다.")
                    except Exception as e:
                        st.error(f"답변 생성 중 오류가 발생했습니다: {str(e)}")
            else:
                st.warning("질문을 입력해주세요.")
    else:
        st.info("처리된 영상이 없습니다. 먼저 영상을 처리해주세요.")


def show_tag_based_question(user_id):
    all_tags = database.get_all_tags()
    selected_tags = st.multiselect("태그 선택", all_tags, key="tag_selector")

    if selected_tags:
        videos = select_videos_by_tags(selected_tags)
        if videos:
            st.write(f"선택된 영상 수: {len(videos)}")
            video_titles = [f"{v['title']} - {v['channel']}" for v in videos]
            st.write("선택된 영상:", ", ".join(video_titles))

            question = st.text_input("질문을 입력하세요")
            if st.button("답변 받기"):
                if question:
                    with st.spinner("답변 생성 중..."):
                        try:
                            video_data = database.get_video_info_from_db([v['video_id'] for v in videos])
                            if video_data:
                                transcripts = [v['transcript'] for v in video_data if 'transcript' in v]
                                response = nlp.generate_response(question, transcripts)
                                display_response(question, response)
                            else:
                                st.error("선택한 영상의 트랜스크립트를 찾을 수 없습니다.")
                        except Exception as e:
                            st.error(f"답변 생성 중 오류가 발생했습니다: {str(e)}")
                else:
                    st.warning("질문을 입력해주세요.")
        else:
            st.warning("선택한 태그에 해당하는 영상가 없습니다.")
    else:
        st.info("태그를 선택하여 영상를 필터링하세요.")


def display_response(question, response):
    st.markdown("### 질문:")
    st.write(question)
    st.divider()
    st.markdown("### 답변:")
    st.write(response)
def select_videos_by_tags(tags):
    return database.get_videos_by_tags(tags)


def show_processed_videos():
    st.header("처리된 영상목록", divider=True)
    user_id = st.session_state.user['_id']
    logger.info(f"User ID: {user_id}")

    # # 오늘 날짜 표시
    # today = datetime.now().date()
    # st.sidebar.info(f"오늘 날짜: {today.strftime('%Y년 %m월 %d일')}")

    # 필터 옵션 (세로 배치)
    st.subheader("필터 옵션")
    col1, col2, col3 = st.columns(3)
    with col1:
        all_tags = database.get_all_tags()
        logger.info(f"All tags: {all_tags}")
        selected_tags = st.multiselect("태그 선택", all_tags)
    with col2:
        today = datetime.now().date()
        date_range = st.date_input("기간 선택", [today, today])
    with col3:
        show_no_tags = st.checkbox("태그 없는 영상만 표시")


    # 필터 적용
    start_date = None
    end_date = None
    if date_range:
        if isinstance(date_range, (list, tuple)):
            if len(date_range) == 2:
                start_date, end_date = date_range
            elif len(date_range) == 1:
                start_date = end_date = date_range[0]
        else:
            start_date = end_date = date_range

        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())

    logger.info(f"Date range: {start_date} to {end_date}")

    # 모든 영상를 가져옴 (필터 적용)
    valid_videos = database.get_user_videos(user_id, selected_tags=selected_tags, start_date=start_date,
                                            end_date=end_date, show_no_tags=show_no_tags)

    logger.info(f"Number of videos retrieved: {len(valid_videos)}")

    if valid_videos:
        for video in valid_videos:
            with st.container():
                # 영상 제목 (카드 형태)
                st.markdown(
                    f'<i class="fab fa-youtube" style="margin-right: 5px; color: red;"></i>📹 <span style="font-size: 20px;">**{video.get("title", "Unknown")}**</span>',
                    unsafe_allow_html=True)

                # 메타데이터 (일렬 배치)
                st.markdown(f"""
                채널명: {video.get('channel', 'Unknown')} | 처리 일자: {video.get('processed_at', datetime.now()).strftime('%Y-%m-%d %H:%M')} | 길이: {video_processing.format_time(video.get('duration', 0))} | 처리된 글자수: {video.get('transcript_length', 0)} 자
                """, unsafe_allow_html=True)
                st.divider()

                # 태그 영역
                tags = video.get('tags', [])
                if tags:
                    st.markdown("##### 태그")
                    for tag in tags:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(tag)
                        with col2:
                            if st.button("삭제", key=f"delete_{video['video_id']}_{tag}"):
                                if delete_tag(video['video_id'], tag):
                                    st.success(f"태그 '{tag}'가 삭제되었습니다.")
                                    time.sleep(1)  # 사용자가 메시지를 볼 수 있도록 잠시 대기
                                    st.rerun()  # 페이지 새로고침
                                else:
                                    st.error("태그 삭제 중 오류가 발생했습니다.")
                else:
                    st.write("태그가 없습니다.")

                # 새 태그 추가 영역
                col1, col2 = st.columns([3, 1])
                with col1:
                    # 동적 키 생성
                    input_key = f"new_tag_{video['video_id']}_{st.session_state.get('tag_input_key', 0)}"
                    new_tag = st.text_input("새 태그 추가", key=input_key, placeholder="새 태그 입력")
                with col2:
                    if st.button("태그 추가", key=f"add_tag_{video['video_id']}"):
                        if new_tag:
                            if database.add_tag_to_video(video['video_id'], new_tag):
                                st.success("태그가 추가되었습니다.")
                                # 입력 필드 키 변경
                                st.session_state['tag_input_key'] = st.session_state.get('tag_input_key', 0) + 1
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.warning("태그를 추가할 수 없습니다. (최대 3개)")
                        else:
                            st.warning("태그를 입력해주세요.")

                st.divider()

                # 버튼 배치 (가로 배치)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("질문하기", key=f"chat_{video['video_id']}"):
                        st.session_state.page = 'chat'
                        st.session_state.selected_video_id = video['video_id']
                        st.rerun()
                with col2:
                    if st.button("전문보기", key=f"full_{video['video_id']}"):
                        st.session_state.page = 'full_transcript'
                        st.session_state.selected_video_id = video['video_id']
                        st.rerun()

            st.markdown("---")  # 영상 사이에 구분선 추가

    else:
        logger.warning("No videos found for the user.")
        st.info("선택한 조건에 맞는 영상이 없습니다.")


def show_chat_page():
    st.header("영상 채팅")
    if st.session_state.selected_video_id:
        # selected_video_id를 리스트로 감싸서 전달
        video_data = database.get_video_info_from_db([st.session_state.selected_video_id])
        if video_data and len(video_data) > 0:
            video = video_data[0]  # 첫 번째 (유일한) 결과를 사용
            st.subheader(f"영상: {video.get('title', 'Unknown')}")

            question = st.text_input("질문을 입력하세요")
            if st.button("답변 받기"):
                if question:
                    with st.spinner("답변 생성 중..."):
                        try:
                            response = nlp.generate_response(question, [video.get('transcript', '')])
                            st.markdown("### 질문:")
                            st.write(question)
                            st.markdown("### 답변:")
                            st.write(response)
                        except Exception as e:
                            st.error(f"답변 생성 중 오류가 발생했습니다: {str(e)}")
                else:
                    st.warning("질문을 입력해주세요.")
        else:
            st.error("선택한 영상의 정보를 찾을 수 없습니다.")
    else:
        st.error("선택된 영상가 없습니다.")

    if st.button("영상 목록으로 돌아가기"):
        st.session_state.page = "view_videos"
        st.session_state.selected_video_id = None
        st.rerun()


def delete_tag(video_id, tag):
    """태그 삭제 함수"""
    try:
        database.remove_tag_from_video(video_id, tag)
        return True
    except Exception as e:
        logger.error(f"태그 삭제 중 오류 발생: {str(e)}")
        return False

def get_valid_videos(user_id):
    all_videos = database.get_user_videos(user_id)
    return [video for video in all_videos if video.get('title') and video.get('channel')]

def show_full_transcript():
    st.header("영상 전체 내용보기")
    if st.session_state.selected_video_id:
        # selected_video_id를 리스트로 감싸서 전달
        video_data = database.get_video_info_from_db([st.session_state.selected_video_id])
        if video_data and len(video_data) > 0:
            video = video_data[0]  # 첫 번째 (유일한) 결과를 사용
            st.markdown(f'<span style="font-size: 24px;">**{video.get("title", "Unknown")}**</span>', unsafe_allow_html=True)
            st.write(f"채널명: {video.get('channel', 'Unknown')}")
            st.write("전문:")
            st.text_area("", value=video.get('transcript', ''), height=400, disabled=True)
        else:
            st.error("선택한 영상의 정보를 찾을 수 없습니다.")
    else:
        st.error("선택된 영상가 없습니다.")

    if st.button("영상 목록으로 돌아가기"):
        st.session_state.page = "view_videos"
        st.session_state.selected_video_id = None
        st.rerun()

def update_processed_videos(user_id):
    st.session_state.processed_videos = database.get_user_videos(user_id)

def show_feedback_form():
    st.header("피드백 남기기")
    user_id = st.session_state.user['_id']
    feedback = st.text_area("서비스에 대한 의견이나 개선 사항을 자유롭게 작성해주세요.")
    if st.button("피드백 제출"):
        if feedback:
            database.save_feedback(user_id, feedback)
            st.success("피드백이 성공적으로 제출되었습니다. 감사합니다!")
        else:
            st.warning("피드백 내용을 입력해주세요.")

def add_tag_callback(video_id, new_tag):
    if new_tag:
        if database.add_tag_to_video(video_id, new_tag):
            st.success("태그가 추가되었습니다.")
            # 입력 필드 초기화
            st.session_state[f"new_tag_{video_id}"] = ""
            time.sleep(1)
            st.rerun()
        else:
            st.warning("태그를 추가할 수 없습니다. (최대 3개)")
    else:
        st.warning("태그를 입력해주세요.")
