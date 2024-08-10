import streamlit as st
from modules import auth, video_processing, database, ui, nlp

def initialize_session_state():
    if 'processed_videos' not in st.session_state:
        st.session_state.processed_videos = []
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'selected_video_id' not in st.session_state:
        st.session_state.selected_video_id = None

def main():
    st.set_page_config(page_title="AskOnTube", page_icon="🎥", layout="wide")
    ui.show_header()

    initialize_session_state()

    if st.session_state.user:
        ui.show_sidebar()

    if st.session_state.page == 'login':
        ui.show_login_form()
    elif st.session_state.page == 'process_video':
        ui.show_video_processing_form()
    elif st.session_state.page == 'ask_question':
        ui.show_question_form()
    elif st.session_state.page == 'view_videos':
        ui.show_processed_videos()
    elif st.session_state.page == 'full_transcript':
        ui.show_full_transcript()
    elif st.session_state.page == 'chat':
        ui.show_chat_page()
    elif st.session_state.page == 'feedback':
        ui.show_feedback_form()

if __name__ == "__main__":
    main()
