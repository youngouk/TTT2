from openai import OpenAI
import google.generativeai as genai
from config import OPENAI_API_KEY, GEMINI_API_KEY
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# OpenAI 클라이언트 초기화
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Gemini API 설정
genai.configure(api_key=GEMINI_API_KEY)

def transcribe_audio(file_path):
    """오디오 파일을 텍스트로 변환"""
    with open(file_path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

def embed_text(text):
    """텍스트를 벡터로 임베딩"""
    response = openai_client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding


def generate_response(query, transcripts):
    """여러 트랜스크립트를 기반으로 질문에 대한 응답 생성"""
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

    relevant_parts = process_multiple_transcripts(query, transcripts)
    combined_transcript = "\n\n".join(relevant_parts)

    prompt = textwrap.dedent(f"""
    다음은 여러 YouTube 비디오의 관련 내용입니다:

    {combined_transcript}

    질문: {query}

    위의 내용을 바탕으로 질문에 답변해주세요. 답변 시 다음 지침을 따라주세요:
    1. 주어진 내용에서 직접적으로 관련된 정보를 찾아 상세하게 답변하세요.
    2. 필요한 경우 풍부한 설명과 예시를 포함하여 답변하세요.
    3. 정보가 부족하거나 관련이 없는 경우, "제공된 내용에는 이 질문에 답할 만한 충분한 정보가 없습니다."라고 명시한 후, 기존 지식을 활용하여 일반적인 수준의 추가 정보를 제공하세요.
    4. 관련 부분을 직접 인용하여 답변의 근거를 제시하세요. 인용 시 큰따옴표를 사용하고 출처를 명시하세요.
    5. 의학적 조언이나 전문적인 내용을 다룰 때는 "영상에서 언급된 바에 따르면"이라는 문구로 시작하고, 추가적인 전문가 상담을 권고하세요.
    6. 긴 답변을 제공하는 경우 마지막 문잔에 주요 포인트를 요약하고, 추가 학습이나 탐구를 위한 제안을 포함하세요.
    7. 답변의 깊이, 양이 구체적으로 명시되지 않은 질문에 대해서는 기본적으로 10줄 이상의 구체적 답변을 하세요.

    답변:
    """)

    try:
        response = model.generate_content(prompt)
        return response.text
    except genai.types.generation_types.BlockedPromptException:
        return "죄송합니다. 이 질문에 대한 응답을 생성할 수 없습니다. 다른 방식으로 질문을 표현해 보시겠습니까?"
    except Exception as e:
        return f"응답 생성 중 오류 발생: {str(e)}"


def process_multiple_transcripts(query, transcripts):
    """여러 트랜스크립트에서 질문과 관련성 높은 부분 선별"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(transcripts + [query])

    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]  # 상위 5개 관련 문서 선택

    relevant_parts = [transcripts[i] for i in related_docs_indices]
    return relevant_parts
