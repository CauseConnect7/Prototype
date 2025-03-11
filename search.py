import openai
import json
import pandas as pd
from pymongo import MongoClient
import numpy as np
from bson.binary import Binary
import streamlit as st
from scipy.spatial.distance import cosine
import requests
from dotenv import load_dotenv
import os
import re

# ----------- Streamlit UI代码 -----------
# 设置页面配置（必须是第一个Streamlit命令）
st.set_page_config(page_title="Organization Matcher", layout="wide")

# 使用缓存装饰器优化数据库连接
@st.cache_resource(ttl=3600)
def init_mongodb_connection():
    """Initialize MongoDB connection and return necessary database and collection objects"""
    try:
        client = MongoClient(os.getenv("MONGODB_URI"))
        client.server_info()  # Test connection
        
        # Initialize databases and collections
        db = client[os.getenv("MONGODB_DB_NAME")]
        collection1 = db[os.getenv("MONGODB_COLLECTION_NONPROFIT")]
        collection2 = db[os.getenv("MONGODB_COLLECTION_FORPROFIT")]
        user_input_db = client[os.getenv("MONGODB_DB_OUTPUT_NAME")]
        profile_collection = user_input_db[os.getenv("MONGODB_COLLECTION_PROFILE")]
        
        return client, db, collection1, collection2, user_input_db, profile_collection
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {str(e)}")
        raise Exception(f"Failed to connect to MongoDB: {str(e)}")

@st.cache_data(ttl=3600)
def get_collection_stats():
    """Get collection statistics"""
    try:
        stats = {
            "nonprofit_count": collection1.count_documents({}),
            "forprofit_count": collection2.count_documents({}),
            "profile_count": profile_collection.count_documents({})
        }
        return stats
    except Exception as e:
        st.error(f"Error getting collection statistics: {str(e)}")
        return None

# 添加清除环境变量缓存的功能
@st.cache_data(ttl=3600)
def reload_env():
    """Reload environment variables"""
    # 清除特定的环境变量
    env_vars = [
        "MATCH_EVALUATION_SYSTEM_PROMPT",
        "MATCH_EVALUATION_PROMPT",
        "MONGODB_URI",
        "MONGODB_DB_NAME",
        "MONGODB_COLLECTION_NONPROFIT",
        "MONGODB_COLLECTION_FORPROFIT",
        "OPENAI_API_KEY"
    ]
    
    # 清除指定的环境变量
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]
    
    # 重新加载.env文件
    load_dotenv(override=True)
    return "Environment variables reloaded"

# ----------- Load Environment Variables -----------
load_dotenv()

# ----------- MongoDB Connection -----------
# Initialize database connection
try:
    client, db, collection1, collection2, user_input_db, profile_collection = init_mongodb_connection()
    # Verify collections are accessible
    stats = get_collection_stats()
    if stats:
        st.success("Successfully connected to database")
except Exception as e:
    st.error(f"Error connecting to database: {str(e)}")
    raise Exception(f"Failed to connect to database: {str(e)}")

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# ----------- 函数定义 -----------
@st.cache_data(ttl=3600)
def generate_ideal_organization(row):
    """Generate 10 organizations based on needs, then filter to 3 based on mission alignment."""
    try:
        # Step 1: Generate 10 potential organizations
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": os.getenv("PROMPT_GEN_ORG_SYSTEM").format(
                    org_type_looking_for=row["Organization looking 1"])},
                {"role": "user", "content": os.getenv("PROMPT_GEN_ORG_USER").format(
                    org_type_looking_for=row["Organization looking 1"],
                    partnership_description=row["Organization looking 2"])}
            ]
        )

        generated_organizations = response['choices'][0]['message']['content'].strip()

        # Step 2: Filter down to the 3 best matches
        filtered_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": os.getenv("PROMPT_FILTER_SYSTEM")},
                {"role": "user", "content": os.getenv("PROMPT_FILTER_USER").format(
                    organization_mission=row["Description"],
                    generated_organizations=generated_organizations)}
            ]
        )

        return filtered_response['choices'][0]['message']['content'].strip()

    except Exception as e:
        st.error(f"Error generating organizations: {str(e)}")
        return ""

# ----------- Define Structured Tagging Steps -----------
step_descriptions = {
    1: os.getenv("TAG_STEP_1"),
    2: os.getenv("TAG_STEP_2"),
    3: os.getenv("TAG_STEP_3"),
    4: os.getenv("TAG_STEP_4"),
    5: os.getenv("TAG_STEP_5"),
    6: os.getenv("TAG_STEP_6")
}

@st.cache_data(ttl=3600)
def generate_fixed_tags(description, audience, total_tags=30, steps=6, tags_per_step=5):
    """Generate structured AI-powered tags following a 6-step format."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # 使用相同的模型
            messages=[
                {"role": "system", "content": os.getenv("PROMPT_TAGS_SYSTEM").format(
                    total_tags=total_tags,
                    steps=steps,
                    tags_per_step=tags_per_step
                )},
                {"role": "user", "content": os.getenv("PROMPT_TAGS_USER").format(
                    total_tags=total_tags,
                    description=description
                )}
            ]
        )
        tags = response['choices'][0]['message']['content'].strip()

        # Convert tags to a list and normalize to exactly `total_tags`
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        tag_list = tag_list[:total_tags]  # Ensure 30 tags

        return ", ".join(tag_list)
    except Exception as e:
        st.error(f"Error generating tags: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_embedding(text):
    """Generate vector embedding using OpenAI."""
    if not text or not isinstance(text, str):
        return None  # Skip invalid data

    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

def find_top_50_matches(embedding, looking_for_type):
    """Find top 50 matching organizations based on embedding similarity."""
    matches = []
    # 确保类型匹配完全一致
    collection = collection1 if looking_for_type.strip() == os.getenv("MONGODB_COLLECTION_NONPROFIT").strip() else collection2
    
    try:
        for org in collection.find({"Embedding": {"$exists": True}}):
            if org.get("Embedding"):
                org_embedding = np.frombuffer(org["Embedding"], dtype=np.float32)
                similarity = 1 - cosine(embedding, org_embedding)
                matches.append((
                    similarity,
                    org.get("Name", "Unknown"),
                    org.get("Description", "No description available"),
                    org.get("URL", "N/A"),
                    org.get("linkedin_description", "No LinkedIn description available"),
                    org.get("linkedin_tagline", "No tagline available"),
                    org.get("linkedin_type", "N/A"),
                    org.get("linkedin_industries", "N/A"),
                    org.get("linkedin_specialities", "N/A"),
                    org.get("linkedin_staff_count", "N/A"),
                    org.get("City", "N/A"),
                    org.get("State", "N/A"),
                    org.get("linkedin_url", "N/A"),
                    org.get("Tag", "No tags available")
                ))
        
        matches.sort(reverse=True)
        return matches[:100]
    except Exception as e:
        st.error(f"Error finding matches: {str(e)}")
        return []

def evaluate_match_with_gpt(org_info, user_info):
    """Use GPT to evaluate match quality and decide whether to keep the match"""
    try:
        # Get prompts from environment variables
        system_prompt = os.getenv("MATCH_EVALUATION_SYSTEM_PROMPT")
        evaluation_prompt = os.getenv("MATCH_EVALUATION_PROMPT")
        
        if not system_prompt or not evaluation_prompt:
            raise ValueError("Required prompts not found in environment variables")

        # Format the evaluation prompt
        try:
            formatted_prompt = evaluation_prompt.format(
                user_name=user_info['Name'],
                user_type=user_info['Type'],
                user_description=user_info['Description'],
                user_target_audience=user_info['Target Audience'],
                user_looking_type=user_info['Organization looking 1'],
                user_partnership_desc=user_info['Organization looking 2'],
                match_name=org_info[1],
                match_description=org_info[2],
                match_linkedin_desc=org_info[4],
                match_tagline=org_info[5],
                match_type=org_info[6],
                match_industry=org_info[7],
                match_specialties=org_info[8],
                match_tags=org_info[13]
            )
        except KeyError as e:
            st.error(f"Missing required field in organization data: {str(e)}")
            return False
        except Exception as e:
            st.error(f"Error formatting prompt: {str(e)}")
            return False

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.3
        )
        
        result = response['choices'][0]['message']['content'].strip().lower()
        return result == 'true'
    except Exception as e:
        st.error(f"Error evaluating match: {str(e)}")
        return False  # Changed to False to be more conservative with errors

def process_matches(tags, looking_for_type, row):
    """处理匹配逻辑，确保返回20个匹配结果"""
    if not tags:
        return []
    
    embedding = get_embedding(tags)
    if embedding is None:
        return []
    
    all_matches = find_top_50_matches(embedding, looking_for_type)
    if not all_matches:
        return []
    
    filtered_matches = []
    remaining_matches = []
    
    with st.spinner("Finding matching organizations..."):
        progress_bar = st.progress(0)
        
        remaining_matches = all_matches.copy()
        
        for i, match in enumerate(all_matches[:30]):
            if evaluate_match_with_gpt(match, row):
                filtered_matches.append(match)
                remaining_matches.remove(match)
            progress_bar.progress((i + 1) / 30)
        
        if len(filtered_matches) < 20:
            remaining_needed = 20 - len(filtered_matches)
            filtered_matches.extend(remaining_matches[:remaining_needed])
    
    return filtered_matches[:20]

def display_matches(filtered_matches, gpt_verified_count):
    # 显示匹配结果标题
    st.subheader("Top 20 Matching Organizations:")
    st.write(f"({gpt_verified_count} AI-verified matches, {20-gpt_verified_count} Similarity-based matches)")
    
    # 初始化评价状态
    if 'match_evaluations' not in st.session_state:
        st.session_state.match_evaluations = {}
    if 'overall_satisfaction' not in st.session_state:
        st.session_state.overall_satisfaction = 3  # 默认值改为3（中间值）
    if 'satisfaction_reason' not in st.session_state:
        st.session_state.satisfaction_reason = ""
    
    # 更新CSS样式，添加评估区域的样式
    st.markdown("""
        <style>
        .match-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .evaluation-section {
            background-color: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
            border: 1px solid #e0e0e0;
        }
        
        .evaluation-title {
            color: #2c3e50;
            font-size: 1.5em;
            margin-bottom: 20px;
            font-weight: 600;
        }
        
        .evaluation-question {
            color: #2d3748;
            font-size: 1em;
            margin: 15px 0 5px 0;
            font-weight: 500;
        }
        
        .evaluation-label {
            color: #666;
            font-size: 0.9em;
            margin: 0;
        }
        
        .stSlider {
            width: 90%;
            margin: 0 auto;
        }
        
        .stSlider > div > div > div {
            padding: 0 !important;
        }
        
        .feedback-section {
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #eee;
        }
        
        .submit-button {
            margin-top: 20px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 使用列布局创建网格
    cols = st.columns(2)
    
    for i, match in enumerate(filtered_matches, 1):
        similarity, name, description, url, linkedin_desc, tagline, type_, industries, specialities, staff_count, city, state, linkedin_url, match_tags = match
        match_type = "AI-Verified Match" if i <= gpt_verified_count else "Similarity Match"
        
        # 在两列之间交替显示卡片
        with cols[i % 2]:
            with st.container():
                st.markdown(f"""
                    <div class="match-card">
                        <div class="match-header">
                            <div class="match-title">{name}</div>
                            <div class="match-type">{match_type}</div>
                        </div>
                        <div class="match-description">{description}</div>
                        <div class="match-details">
                            <p><strong>Location:</strong> {city}, {state}</p>
                            <p><strong>Type:</strong> {type_}</p>
                            <p><strong>Staff Count:</strong> {staff_count}</p>
                            <p><strong>Industries:</strong> {industries}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # 使用expander显示额外信息
                with st.expander("Additional Information"):
                    # Logo 部分
                    try:
                        collection = collection1 if looking_for_type.strip() == os.getenv("MONGODB_COLLECTION_NONPROFIT").strip() else collection2
                        org_data = collection.find_one({"Name": name})
                        if org_data and "linkedin_logo" in org_data:
                            logo_url = org_data["linkedin_logo"]
                            st.markdown(f"""
                                <div class="logo-container">
                                    <img src="{logo_url}" alt="{name} logo">
                                </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.write("Logo not available")

                    # 使用两列布局显示详细信息
                    st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                    
                    # 左列
                    st.markdown('<div class="info-section">', unsafe_allow_html=True)
                    st.markdown("**Organization Links**")
                    if linkedin_url:
                        st.write(f"[LinkedIn Profile]({linkedin_url})")
                    if url and url != "N/A":
                        st.write(f"[Website]({url})")
                    
                    st.markdown("**Basic Details**")
                    st.write(f"**Type:** {type_}")
                    st.write(f"**Location:** {city}, {state}")
                    st.write(f"**Staff Count:** {staff_count}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 右列
                    st.markdown('<div class="info-section">', unsafe_allow_html=True)
                    st.markdown("**LinkedIn Information**")
                    st.write(f"**Tagline:** {tagline}")
                    st.write(f"**Industries:** {industries}")
                    st.write(f"**Specialties:** {specialities}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 底部显示完整描述和标签
                    st.markdown("**Full LinkedIn Description**")
                    st.write(linkedin_desc)
                    
                    st.markdown("**Tags & Keywords**")
                    st.write(match_tags)
                
                # 修改这部分，添加索引到key中以确保唯一性
                st.markdown('<div class="match-evaluation">', unsafe_allow_html=True)
                st.session_state.match_evaluations[name] = st.radio(
                    "Is this a good match?",
                    ["Yes", "No", "Maybe"],
                    key=f"eval_{i}_{name}",  # 添加索引i来确保key的唯一性
                    horizontal=True,
                    index=0 if st.session_state.match_evaluations.get(name, "Maybe") == "Yes" else 
                          1 if st.session_state.match_evaluations.get(name, "Maybe") == "No" else 2
                )
                st.markdown('</div>', unsafe_allow_html=True)

    # 在所有组织卡片之后添加整体评价部分
    st.markdown("""
        <div class='evaluation-section'>
            <div class='evaluation-title'>Overall Evaluation</div>
        </div>
    """, unsafe_allow_html=True)

    # 创建评估问题
    evaluation_questions = {
        "prototype_enhancement": {
            "question": "Does our prototype enhance the matching process compared to your previous workflow?",
            "key": "prototype_rating",
            "labels": {1: "Not that Much", 5: "Very"}
        },
        "overall_satisfaction": {
            "question": "How would you rate your overall experience with this survey?",
            "key": "satisfaction_rating",
            "labels": {1: "Not Satisfied", 5: "Extremely Satisfied"}
        },
        "process_intuitive": {
            "question": "Was the whole process intuitive and easy to answer?",
            "key": "intuitive_rating",
            "labels": {1: "Not Intuitive", 5: "Very Intuitive"}
        }
    }

    # 为每个问题创建滑动条
    for key, data in evaluation_questions.items():
        if key not in st.session_state:
            st.session_state[key] = 3

        st.markdown(f"<div class='evaluation-question'>{data['question']}</div>", unsafe_allow_html=True)
        
        # 创建更紧凑的三列布局
        col1, col2, col3 = st.columns([1.5, 7, 1.5])
        
        # 显示左侧标签
        with col1:
            st.markdown(f"<div class='evaluation-label' style='text-align: left;'>{data['labels'][1]}</div>", unsafe_allow_html=True)
        
        # 显示滑动条
        with col2:
            st.session_state[key] = st.slider(
                data['question'],
                min_value=1,
                max_value=5,
                value=st.session_state[key],
                key=data['key'],
                label_visibility="collapsed"
            )
        
        # 显示右侧标签
        with col3:
            st.markdown(f"<div class='evaluation-label' style='text-align: right;'>{data['labels'][5]}</div>", unsafe_allow_html=True)

    # 添加反馈文本框
    st.markdown("<div class='feedback-section'>", unsafe_allow_html=True)
    st.markdown("<div class='evaluation-question'>Additional Feedback</div>", unsafe_allow_html=True)
    st.session_state.satisfaction_reason = st.text_area(
        "Please provide any additional feedback:",
        value=st.session_state.get('satisfaction_reason', ''),
        height=100,
        key="feedback_text",
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 添加提交按钮
    st.markdown("<div class='submit-button'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_disabled = not st.session_state.user_email
        if st.button("Submit All Evaluations", 
                    use_container_width=True, 
                    disabled=submit_disabled,
                    help="Please provide your email address to submit" if submit_disabled else None):
            if st.session_state.get('current_row'):
                # 验证邮箱格式
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, st.session_state.user_email):
                    st.error("Please enter a valid email address")
                    return
                
                # 准备要存储的数据
                row = st.session_state['current_row'].copy()
                if '_id' in row:
                    del row['_id']
                
                # 添加评价数据
                row.update({
                    "Match Evaluations": st.session_state.match_evaluations,
                    "Matched Organizations": [match[1] for match in filtered_matches[:20]],
                    "Prototype Enhancement Rating": st.session_state.prototype_enhancement,
                    "Overall Satisfaction": st.session_state.overall_satisfaction,
                    "Process Intuitive Rating": st.session_state.process_intuitive,
                    "Satisfaction Reason": st.session_state.satisfaction_reason,
                    "User Email": st.session_state.user_email
                })
                
                try:
                    store_user_input_to_db(row)
                    st.success("Thank you! Your evaluations have been saved.")
                    # 重置评价状态
                    for key in evaluation_questions:
                        st.session_state[key] = 3
                    st.session_state.satisfaction_reason = ""
                except Exception as e:
                    st.error(f"Error saving evaluations: {str(e)}")

def store_user_input_to_db(user_data):
    """Store user input data to MongoDB User Input database in Profile collection."""
    try:
        # Ensure database connection exists
        if user_input_db is None or profile_collection is None:
            raise Exception("Database connection not initialized")
            
        # Validate database connection
        try:
            # Test database connection
            user_input_db.command('ping')
        except Exception as e:
            raise Exception(f"Invalid database connection: {str(e)}")
            
        # Insert data into Profile collection
        result = profile_collection.insert_one(user_data)
        
        if result and result.inserted_id:
            st.success(f"User feedback successfully saved to database! (Document ID: {result.inserted_id})")
            return True
        else:
            st.error("Failed to save user feedback: Could not get insertion ID")
            return False
            
    except Exception as e:
        st.error(f"Error saving user feedback data: {str(e)}")
        return False

# ----------- Streamlit UI代码 -----------
# 创建主标题和介绍
st.markdown("""
    <div style='background-color: #f0f9ff; padding: 20px; border-radius: 10px; margin-bottom: 30px; border: 1px solid #90cdf4;'>
        <h3 style='color: #2b6cb0; margin-bottom: 15px;'>Thank you for participating in our research!</h3>
        <p style='color: #4a5568; margin-bottom: 15px;'>To receive your Amazon Gift Card, please provide your email address below. After completing the organization evaluation, remember to click the "Submit All Evaluations" button at the bottom of the page.</p>
    </div>
""", unsafe_allow_html=True)

# 在蓝色框内添加邮箱输入
with st.container():
    st.session_state.user_email = st.text_input(
        "Your Email Address *",
        value=st.session_state.get('user_email', ''),
        key="email_input",
        help="Please provide your email to receive the Amazon Gift Card"
    )

st.title("Organization Partnership Matcher")
st.markdown("""
This tool helps you find potential partnership organizations that align with your values and goals.
Please fill in the information about your organization below.
""")

# 初始化所有session_state变量
if 'form_data' not in st.session_state:
    st.session_state['form_data'] = {
        'name': '',
        'type': 'Non Profit',
        'description': '',
        'category': '',
        'audience': '',
        'state': '',
        'city': '',
        'website': '',
        'looking_for_type': 'Non Profit',
        'looking_for_desc': ''
    }

if 'satisfaction_score' not in st.session_state:
    st.session_state['satisfaction_score'] = 5
if 'satisfaction_reason' not in st.session_state:
    st.session_state['satisfaction_reason'] = ''
if 'filtered_matches' not in st.session_state:
    st.session_state['filtered_matches'] = []
if 'generated_orgs' not in st.session_state:
    st.session_state['generated_orgs'] = ''
if 'tags' not in st.session_state:
    st.session_state['tags'] = ''
if 'feedback_submitted' not in st.session_state:
    st.session_state['feedback_submitted'] = False
if 'current_row' not in st.session_state:
    st.session_state['current_row'] = None

# 初始化新的session state变量
if 'difficulty_rating' not in st.session_state:
    st.session_state.difficulty_rating = 3

# 初始化session state变量
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""

# 创建表单以收集所有输入
with st.form(key='organization_form'):
    # 第一部分：组织档案
    st.header("Section 1: Build Your Organization Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        org_name = st.text_input("Organization Name*", 
                                value=st.session_state['form_data']['name'],
                                key="name_input")
        org_type = st.selectbox("Organization Type*",
                               ["Non Profit", "For-Profit"],
                               index=0 if st.session_state['form_data']['type'] == "Non Profit" else 1,
                               key="type_input")
        org_description = st.text_area("Organization Mission Statement*",
                                     value=st.session_state['form_data']['description'],
                                     key="description_input")
    
    with col2:
        org_category = st.text_area("Core Values*",
                                   value=st.session_state['form_data']['category'],
                                   key="category_input")
        target_audience = st.text_area("Target Audience*",
                                     value=st.session_state['form_data']['audience'],
                                     key="audience_input")
    
    # 位置信息
    col3, col4 = st.columns(2)
    with col3:
        state = st.text_input("State",
                             value=st.session_state['form_data']['state'],
                             key="state_input")
    with col4:
        city = st.text_input("City",
                            value=st.session_state['form_data']['city'],
                            key="city_input")
    
    website = st.text_input("Website URL",
                           value=st.session_state['form_data']['website'],
                           key="website_input")
    
    # 第二部分：匹配过程
    st.header("Section 2: Start the Matching Process")
    
    looking_for_type = st.selectbox("Organization Type Looking For*",
                                   ["Non Profit", "For-Profit"],
                                   index=0 if st.session_state['form_data']['looking_for_type'] == "Non Profit" else 1,
                                   key="looking_type_input")
    
    looking_for_description = st.text_area("What Kind of Organization Are You Looking For?*",
                                         value=st.session_state['form_data']['looking_for_desc'],
                                         key="looking_desc_input")
    
    # 提交按钮
    submitted = st.form_submit_button("Find Matching Organizations")

# 处理表单提交
if submitted:
    # 更新session_state中的表单数据
    st.session_state['form_data'].update({
        'name': org_name,
        'type': org_type,
        'description': org_description,
        'category': org_category,
        'audience': target_audience,
        'state': state,
        'city': city,
        'website': website,
        'looking_for_type': looking_for_type,
        'looking_for_desc': looking_for_description
    })
    
    if not all([org_name, org_type, org_description, org_category, target_audience, looking_for_type, looking_for_description]):
        st.error("Please fill in all required fields marked with *")
    else:
        row = {
            "Name": org_name,
            "Type": org_type,
            "Description": f"Organization Mission: {org_description}\n\nCore Values: {org_category}\n\nTarget Audience: {target_audience}",
            "State": state,
            "City": city,
            "URL": website,
            "Organization looking 1": looking_for_type,
            "Organization looking 2": looking_for_description,
            "Target Audience": target_audience
        }
        
        with st.spinner("Finding matching organizations..."):
            st.session_state['generated_orgs'] = generate_ideal_organization(pd.Series(row))
            st.session_state['tags'] = generate_fixed_tags(st.session_state['generated_orgs'], row["Target Audience"])
            
            if st.session_state['tags']:
                st.session_state['filtered_matches'] = process_matches(st.session_state['tags'], looking_for_type, row)
                st.session_state['current_row'] = row
                st.session_state['feedback_submitted'] = False

# 直接显示匹配结果
if st.session_state['filtered_matches']:
    display_matches(st.session_state['filtered_matches'], len(st.session_state['filtered_matches']))

# Debug output for environment variables
with st.expander("Environment Variables Status"):
    for var in ["MATCH_EVALUATION_SYSTEM_PROMPT", "MATCH_EVALUATION_PROMPT"]:
        value = os.getenv(var)
        st.write(f"{var}: {'Found' if value else 'Missing'}")

# Verify required environment variables
required_env_vars = [
    "OPENAI_API_KEY",
    "MONGODB_URI",
    "MONGODB_DB_NAME"
]

# Check if all required environment variables are present
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
