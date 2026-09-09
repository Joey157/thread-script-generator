import streamlit as st

import json
import datetime

HISTORY_FILE = "history.json"

def save_session_history(profile, selected_display, top_hooks, generated_scripts, long_text):
    record = {
        "id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "situation": profile.get('situation', ''),
        "long_text": long_text,
        "mode": selected_display,
        "scripts": [{"style": top_hooks[i].get("hook_style", ""), "script": generated_scripts[i]} for i in range(len(generated_scripts))]
    }
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except:
            pass
    history.insert(0, record)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


import pandas as pd
import google.genai as genai
import faiss
import numpy as np
import os
import glob
import json
from PIL import Image

# -------------------------------------------------------------------
# 1. 기본 설정 및 UI 레이아웃
# -------------------------------------------------------------------
st.set_page_config(page_title="쓰레드 쇼핑 대본 생성기", layout="wide")

def check_password():
    """Returns `True` if the user had the correct password."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.markdown("## 🔒 앱 접근 권한이 필요합니다")
        password = st.text_input("비밀번호를 입력하세요", type="password")
        if st.button("확인"):
            # Streamlit Secrets에 저장된 비밀번호와 비교 (로컬 테스트용 기본값 '1234')
            correct_password = st.secrets.get("APP_PASSWORD", "1234")
            if password == correct_password:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("😕 비밀번호가 틀렸습니다.")
        return False
    return True

if not check_password():
    st.stop()

st.title("🔥 쓰레드(Threads) 특화 쇼핑 대본 생성기")

# API 키 입력 (사이드바)
with st.sidebar:
    
    st.header("📌 메뉴")
    menu_mode = st.radio("모드 선택", ["🚀 대본 생성기", "📚 히스토리 보관함"])
    st.markdown("---")
    st.header("⚙️ 설정")

    # 브라우저 세션에만 임시 저장 (파일로 저장하지 않음)
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
        
    api_key_input = st.text_input("Gemini API Key", type="password", value=st.session_state["api_key"])
    
    if api_key_input:
        st.session_state["api_key"] = api_key_input
        os.environ["GOOGLE_API_KEY"] = api_key_input
        os.environ["GEMINI_API_KEY"] = api_key_input
    
    api_key = api_key_input
    st.markdown("---")
    st.markdown("API 키 발급은 [Google AI Studio](https://aistudio.google.com/)에서 가능합니다.")

# DB 폴더 설정
DB_FOLDER = "db_accounts"
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

# CSV 파일 목록 스캔 함수
def get_csv_options():
    csv_files = glob.glob(f"{DB_FOLDER}/*.csv")
    return [os.path.basename(f) for f in csv_files]

# 사이드바에 새로고침 버튼 추가
with st.sidebar:
    if st.button("🔄 CSV 파일 목록 새로고침", use_container_width=True):
        st.rerun()

mode_options = get_csv_options()

# 좌우 분할 (1:1 비율)

def extract_profiling(media_list, text_context):
    client = genai.Client()
    contents = []
    prompt = f"""
    당신은 쓰레드(Threads) 떡상 알고리즘을 분석하는 마케팅 프로파일러입니다.
    제공된 미디어(이미지/영상)와 사용자가 입력한 상세 소구점 텍스트를 종합적으로 분석하여 아래 정보를 JSON 형태로 반환하세요.
    - situation: 이 제품을 사용할 만한 구체적인 상황 묘사
    - target_lack: 타겟 고객이 현재 겪고 있는 불편함이나 결핍
    - visual_weapon: 미디어에서 돋보이는 시각적 무기/특징

    입력 텍스트:
    {text_context}
    """
    contents.append(prompt)
    uploaded_genai_files = []
    
    for item in media_list:
        if isinstance(item, Image.Image):
            contents.append(item)
        elif isinstance(item, dict) and item.get("type") == "video":
            video_file = client.files.upload(file=item["path"])
            import time
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = client.files.get(name=video_file.name)
            contents.append(video_file)
            uploaded_genai_files.append(video_file)
            
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=contents
        )
        for f in uploaded_genai_files:
            client.files.delete(name=f.name)
            
        import json
        text = response.text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except Exception as e:
        for f in uploaded_genai_files:
            try:
                client.files.delete(name=f.name)
            except:
                pass
        st.error(f"프로파일링 실패: {e}")
        return None

def get_embeddings_batch(texts, batch_size=20):
    """텍스트 리스트를 벡터로 변환 (배치 처리로 429 에러 방지 및 재시도 로직 포함)"""
    client = genai.Client()
    all_embeddings = []
    import time
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        contents = [{'role': 'user', 'parts': [{'text': str(t)}]} for t in batch]
        
        retries = 3
        while retries > 0:
            try:
                if i > 0:
                    time.sleep(2)
                result = client.models.embed_content(
                    model="models/gemini-embedding-2",
                    contents=contents
                )
                for emb in result.embeddings:
                    all_embeddings.append(emb.values)
                break
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise e
                time.sleep(10)  # 429 에러 발생 시 10초 대기 후 재시도
    return all_embeddings

def get_query_embedding(text):
    client = genai.Client()
    result = client.models.embed_content(
        model="models/gemini-embedding-2",
        contents=text
    )
    return result.embeddings[0].values

def search_hooks(csv_path, query_text, top_k=5):
    """FAISS를 이용해 가장 유사한 원본 뼈대 검색"""
    df = pd.read_csv(csv_path)
    
    # 신규 포맷(Main Text, Comment Text) 및 구버전(Original Text) 호환 처리
    if "Main Text" in df.columns and "Comment Text" in df.columns:
        texts_for_search = (df["Main Text"].fillna("") + " " + df["Comment Text"].fillna("")).tolist()
        main_texts = df["Main Text"].fillna("").tolist()
        comment_texts = df["Comment Text"].fillna("").tolist()
    elif "Original Text" in df.columns:
        texts_for_search = df["Original Text"].fillna("").tolist()
        main_texts = texts_for_search
        comment_texts = [""] * len(texts_for_search)
    else:
        st.error("CSV 파일에 필요한 열(Main Text, Comment Text)이 없습니다.")
        return []
    
    hook_styles = df["Hook Style"].tolist() if "Hook Style" in df.columns else ["일반"] * len(texts_for_search)
    
    # 텍스트 임베딩 생성 (캐싱 적용)
    cache_file = csv_path + ".npy"
    if os.path.exists(cache_file):
        emb_matrix = np.load(cache_file)
    else:
        # 캐시가 없으면 API 호출
        embeddings = get_embeddings_batch(texts_for_search, batch_size=20)
        emb_matrix = np.array(embeddings).astype('float32')
        np.save(cache_file, emb_matrix)
    
    # FAISS 인덱스 생성
    dimension = emb_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(emb_matrix)
    
    # 쿼리 임베딩 및 검색
    q_emb = np.array([get_query_embedding(query_text)]).astype('float32')
    distances, indices = index.search(q_emb, min(top_k, len(texts_for_search)))
    
    results = []
    for i in indices[0]:
        results.append({
            "hook_style": hook_styles[i],
            "main_text": main_texts[i],
            "comment_text": comment_texts[i]
        })
    return results

def generate_script(profile_data, original_hook):
    """[AI 2] 원본 뼈대의 패턴을 이식하여 최종 대본(본문/댓글) 생성"""
    client = genai.Client()
    prompt = f"""
    당신은 '쓰레드(Threads)' 생태계에 완벽히 최적화된 카피라이터입니다.
    아래 [원본 뼈대]가 가진 특유의 말투, 어그로 방식, 줄바꿈 리듬, 감정선을 '그대로 흉내내어' 새로운 제품의 대본을 작성하세요.
    
    [원본 뼈대]
    스타일: {original_hook['hook_style']}
    본문 텍스트: {original_hook['main_text']}
    댓글 텍스트: {original_hook['comment_text']}
    
    [새로운 제품 프로필]
    - 상황: {profile_data.get('situation', '')}
    - 타겟 결핍: {profile_data.get('target_lack', '')}
    - 시각적 무기: {profile_data.get('visual_weapon', '')}
    
    [작성 규칙]
    1. 원본 뼈대의 Vibe(느낌)를 완벽하게 복제할 것.
    2. 본문과 댓글을 분리하여 작성할 것.
    3. 본문은 정보 제공보다는 '공감'과 '후킹' 위주로 쓰고, "👇 정보는 타래(댓글)에!" 라는 뉘앙스로 마무리할 것.
    4. 댓글은 [광고 표기(선택적)], [핵심 요약 3줄(Bullet)], [👉 여기에 쿠팡 링크 삽입] 구조를 따를 것.
    
    [출력 형식]
    [본문]
    (여기에 본문 내용 작성)
    
    [댓글]
    (여기에 댓글 내용 작성)
    """
    response = client.models.generate_content(model='gemini-3.5-flash', contents=prompt)
    return response.text

# -------------------------------------------------------------------
# 3. 좌측 탭 (입력)
# -------------------------------------------------------------------
if menu_mode == "🚀 대본 생성기":
    col1, col2 = st.columns(2)
    with col1:
        st.header("1. 입력 (Input)")
    
        uploaded_files = st.file_uploader("📸 제품 사진/영상 업로드 (여러 개 가능)", type=["png", "jpg", "jpeg", "webp", "mp4", "mov"], accept_multiple_files=True)
        media_list = []
    
        if uploaded_files:
            with st.expander("🖼️ 업로드된 미디어 미리보기", expanded=False):
                # 미디어를 3열 격자로 작게 표시
                preview_cols = st.columns(3)
                for idx, f in enumerate(uploaded_files):
                    col = preview_cols[idx % 3]
                    if f.type.startswith("image"):
                        img = Image.open(f)
                        media_list.append(img)
                        col.image(img, use_container_width=True)
                    elif f.type.startswith("video"):
                        import uuid
                        # 한글 파일명으로 인한 API 헤더 인코딩 에러(UnicodeEncodeError) 방지를 위해 영문/숫자 난수로 임시 파일명 생성
                        temp_path = f"temp_{uuid.uuid4().hex}.mp4"
                        with open(temp_path, "wb") as out_f:
                            out_f.write(f.read())
                        media_list.append({"type": "video", "path": temp_path})
                        col.video(f)
        
        long_text = st.text_area("📝 상세 소구점 입력 (상세페이지 내용 복붙 환영!)", height=200, 
                                 placeholder="예: 이 로봇청소기는 머리카락 엉킴 방지 브러시가 있고, 물걸레 자동 세척 기능이 있습니다...")
    
    # 파일명 매핑 딕셔너리
        FRIENDLY_NAMES = {
            "PDF1_shopping_note1.csv": "1. 쇼핑노트 계정 스타일(프로필 링크형)",
            "PDF2_various_accounts.csv": "2. 조회수 터진 게시물 스타일",
            "PDF3_jjune713.csv": "3. 쮼713 계정 스타일(레시피 전문)",
        "PDF4_newnew2605.csv": "4. newnew2605 계정 스타일(운동 미용 전문)",
        "PDF5_strawberry7912.csv": "5. strawberry7912 계정 스타일(만능 최강자)"
        }
    
        selected_mode = None
        if not mode_options:
            st.warning("⚠️ `db_accounts` 폴더에 CSV 파일이 없습니다.")
        else:
            # 친숙한 이름으로 변환하여 표시
            display_options = [FRIENDLY_NAMES.get(opt, opt) for opt in mode_options]
            selected_display = st.radio("🎯 벤치마킹 모드 선택", display_options)
            # 선택된 친숙한 이름으로부터 다시 원본 파일명 추출
            selected_mode = mode_options[display_options.index(selected_display)]
        
        generate_btn = st.button("🚀 대본 생성하기", use_container_width=True, type="primary")

    # -------------------------------------------------------------------
    # 4. 우측 탭 (출력)
    # -------------------------------------------------------------------
    with col2:
        st.header("2. 출력 (Output)")
    
        if generate_btn:
            if not api_key:
                st.error("좌측 사이드바에 Gemini API Key를 먼저 입력해주세요!")
            elif not long_text:
                st.error("상세 소구점을 입력해주세요!")
            elif not selected_mode:
                st.error("벤치마킹 모드를 선택해주세요!")
            else:
                with st.status("🚀 대본 생성 프로세스 실행 중...", expanded=True) as status:
                    st.write("👀 1. 미디어 및 텍스트 프로파일링 중 (AI 분석)...")
                    profile = extract_profiling(media_list, long_text)
                
                    if profile:
                        st.write("✅ 프로파일링 완료!")
                    
                        csv_path = os.path.join(DB_FOLDER, selected_mode)
                        query = f"{profile.get('situation', '')} {profile.get('target_lack', '')} {profile.get('visual_weapon', '')}"
                        st.write(f"🔍 2. '{selected_display}'에서 유사 패턴 검색 중...")
                        top_hooks = search_hooks(csv_path, query, top_k=5)
                    
                        if top_hooks:
                            st.write("✅ 유사 떡상 패턴 5개 검색 완료!")
                            st.write("✍️ 3. 개별 패턴에 맞춘 5종 대본 동시 작성 중... (약 5초 소요)")
                        
                            import concurrent.futures
                            import time
                        
                            generated_scripts = [None] * 5
                        
                            def fetch_script(index, hook):
                                # API Rate Limit(무료 과부하) 방지를 위해 약간의 딜레이 부여
                                time.sleep(index * 1.0)
                                return generate_script(profile, hook)
                            
                            # 5개의 쓰레드를 띄워서 동시에 구글 서버에 요청
                            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                                future_to_index = {executor.submit(fetch_script, i, hook): i for i, hook in enumerate(top_hooks)}
                                for future in concurrent.futures.as_completed(future_to_index):
                                    idx = future_to_index[future]
                                    try:
                                        generated_scripts[idx] = future.result()
                                    except Exception as e:
                                        generated_scripts[idx] = f"생성 중 오류 발생: {e}"
                            
                            status.update(label="🚀 모든 대본 생성 완료!", state="complete", expanded=False)
                            save_session_history(profile, selected_display, top_hooks, generated_scripts, long_text)
                        
                            # 4. 결과 출력
                            with st.expander("AI 분석 결과 보기"):
                                st.json(profile)
                            
                            st.subheader("🎉 맞춤형 대본 5종")

                            st.markdown("---")
                            download_content = f"입력 소구점: {long_text}\n\n"
                            for i, script in enumerate(generated_scripts):
                                style = top_hooks[i].get('hook_style', '')
                                download_content += f"=== 대본 {i+1} ({style}) ===\n"
                                download_content += f"{script}\n\n"
                            
                            st.download_button(
                                label="📥 생성된 5개 대본 한 번에 다운로드 (.txt)",
                                data=download_content.encode('utf-8-sig'),
                                file_name=f"대본생성결과_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                            st.markdown("---")
                            tabs = st.tabs([f"대본 {i+1} ({h['hook_style']})" for i, h in enumerate(top_hooks)])
                        
                            for i, tab in enumerate(tabs):
                                with tab:
                                    hook = top_hooks[i]
                                    st.markdown("#### 💡 영감을 준 원본 뼈대")
                                    st.info(f"**본문:** {hook.get('main_text', '')}\n\n    **댓글:** {hook.get('comment_text', '')}")
                                
                                    st.markdown("#### ✨ 생성된 대본")
                                    final_script = generated_scripts[i]
                                
                                    # 본문과 댓글 분리 표시 (간단한 파싱)
                                    if "[댓글]" in final_script:
                                        parts = final_script.split("[댓글]")
                                        main_post = parts[0].replace("[본문]", "").strip()
                                        comment_post = parts[1].strip()
                                    
                                        st.text_area("📝 본문용 텍스트 (클릭 후 Ctrl+A, Ctrl+C로 복사)", main_post, height=150, key=f"main_{i}")
                                        st.text_area("💬 댓글용 텍스트", comment_post, height=150, key=f"comment_{i}")
                                    else:
                                        st.text_area("결과물", final_script, height=300, key=f"full_{i}")

elif menu_mode == "📚 히스토리 보관함":
    st.header("📚 이전 생성 대본 히스토리")
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            
            if not history:
                st.info("아직 저장된 히스토리가 없습니다.")
            else:
                for idx, record in enumerate(history):
                    with st.expander(f"🕒 {record['timestamp']} | 상황: {record['situation'][:30]}... | 벤치마킹: {record['mode']}", expanded=(idx==0)):
                        st.markdown(f"**📝 입력 소구점:** {record.get('long_text', '')}")

                        download_content = f"입력 소구점: {record.get('long_text', '')}\n\n"
                        for i, s in enumerate(record['scripts']):
                            download_content += f"=== 대본 {i+1} ({s['style']}) ===\n"
                            download_content += f"{s['script']}\n\n"
                        
                        st.download_button(
                            label="📥 이 대본 모음 다운로드 (.txt)",
                            data=download_content.encode('utf-8-sig'),
                            file_name=f"대본생성결과_{record['timestamp'].replace(':', '').replace(' ', '_').replace('-', '')}.txt",
                            mime="text/plain",
                            key=f"dl_{idx}"
                        )
                        tabs = st.tabs([f"대본 {i+1} ({s['style']})" for i, s in enumerate(record['scripts'])])
                        for i, tab in enumerate(tabs):
                            with tab:
                                final_script = record['scripts'][i]['script']
                                if "[댓글]" in final_script:
                                    parts = final_script.split("[댓글]")
                                    main_post = parts[0].replace("[본문]", "").strip()
                                    comment_post = parts[1].strip()
                                    st.text_area(f"본문 (히스토리 {idx}_{i})", main_post, height=150)
                                    st.text_area(f"댓글 (히스토리 {idx}_{i})", comment_post, height=150)
                                else:
                                    st.text_area(f"결과물 (히스토리 {idx}_{i})", final_script, height=300)
        except Exception as e:
            st.error(f"히스토리 로드 중 에러: {e}")
    else:
        st.info("아직 저장된 히스토리가 없습니다.")
