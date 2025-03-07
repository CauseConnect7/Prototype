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
    # Select the correct collection based on organization type
    collection = collection1 if looking_for_type.strip() == "Non Profit" else collection2
    
    try:
        st.write(f"Searching in {looking_for_type} database...")
        for org in collection.find({"Embedding": {"$exists": True}}):
            if org.get("Embedding"):
                # Convert from BSON Binary to numpy array
                org_embedding = np.frombuffer(org["Embedding"], dtype=np.float32)
                # Calculate similarity
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
        st.write(f"Found {len(matches)} potential matches")
        return matches[:50]  # Return top 50
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
    
    # 获取前30个相似度匹配
    all_matches = find_top_50_matches(embedding, looking_for_type)
    if not all_matches:
        return []
    
    st.write("Using AI to analyze match quality in depth...")
    filtered_matches = []
    remaining_matches = []
    
    with st.spinner("Analyzing matches..."):
        progress_bar = st.progress(0)
        
        # 首先将所有匹配存入remaining_matches
        remaining_matches = all_matches.copy()
        
        # 对前30个进行GPT评估
        for i, match in enumerate(all_matches[:30]):
            if evaluate_match_with_gpt(match, row):
                filtered_matches.append(match)
                remaining_matches.remove(match)  # 从remaining_matches中移除已匹配的
            progress_bar.progress((i + 1) / 30)
        
        # 确保总是返回20个匹配
        if len(filtered_matches) < 20:
            # 计算需要补充的数量
            remaining_needed = 20 - len(filtered_matches)
            # 从remaining_matches中取出需要的数量（已经按相似度排序）
            filtered_matches.extend(remaining_matches[:remaining_needed])
    
    return filtered_matches[:20]  # 确保只返回20个

def display_matches(filtered_matches, gpt_verified_count):
    """Display matching results with company logos"""
    st.subheader("Top 20 Matching Organizations:")
    st.write(f"({gpt_verified_count} AI-verified matches, {20-gpt_verified_count} Similarity-based matches)")
    
    for i, match in enumerate(filtered_matches, 1):
        similarity, name, description, url, linkedin_desc, tagline, type_, industries, specialities, staff_count, city, state, linkedin_url, match_tags = match
        match_type = "AI-Verified Match" if i <= gpt_verified_count else "Similarity Match"
        
        with st.expander(f"{i}. {name} ({match_type}, Similarity: {match[0]:.2%})"):
            # Create three-column layout
            col1, col2, col3 = st.columns([1, 2, 2])
            
            with col1:
                # Try to get logo URL from MongoDB
                try:
                    collection = collection1 if looking_for_type.strip() == os.getenv("MONGODB_COLLECTION_NONPROFIT").strip() else collection2
                    org_data = collection.find_one({"Name": name})
                    if org_data and "linkedin_logo" in org_data:
                        logo_url = org_data["linkedin_logo"]
                        st.image(logo_url, width=150)
                    else:
                        st.write("No logo available")
                except Exception as e:
                    st.write("Unable to load logo")
            
            with col2:
                st.write("**Basic Information**")
                st.write(f"Description: {description}")
                st.write(f"Website: {url}")
                st.write(f"Location: {city}, {state}")
                st.write("**Tags**")
                st.write(f"{match_tags}")
            
            with col3:
                st.write("**LinkedIn Information**")
                if linkedin_url:
                    st.write(f"[LinkedIn Profile]({linkedin_url})")
                st.write(f"LinkedIn Description: {linkedin_desc}")
                st.write(f"Tagline: {tagline}")
                st.write(f"Type: {type_}")
                st.write(f"Industries: {industries}")
                st.write(f"Specialties: {specialities}")
                st.write(f"Staff Count: {staff_count}")

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

# 显示匹配结果
if st.session_state['generated_orgs']:
    st.subheader("Suggested Organizations:")
    st.write(st.session_state['generated_orgs'])

if st.session_state['tags']:
    st.subheader("Generated Tags:")
    st.write(st.session_state['tags'])

if st.session_state['filtered_matches']:
    display_matches(st.session_state['filtered_matches'], len(st.session_state['filtered_matches']))

    # 显示满意度评分系统
    st.subheader("Rate Your Satisfaction:")
    with st.form(key='satisfaction_form'):
        satisfaction_score = st.slider("Satisfaction Score (1-10)", 1, 10, st.session_state['satisfaction_score'])
        satisfaction_reason = st.text_area("Reason for your rating:", st.session_state['satisfaction_reason'])

        # 提交按钮
        submit_button = st.form_submit_button(label='Submit Rating')

        if submit_button and not st.session_state['feedback_submitted']:
            # 在存储数据之前删除_id字段以避免重复键错误
            row = st.session_state['current_row'].copy()
            if '_id' in row:
                del row['_id']

            # 将评分数据与用户输入数据合并存储到数据库
            row.update({
                "Satisfaction Score": satisfaction_score,
                "Satisfaction Reason": satisfaction_reason,
                "Matched Organizations": [match[1] for match in st.session_state['filtered_matches'][:20]]
            })
            
            try:
                store_user_input_to_db(row)
                st.session_state['feedback_submitted'] = True
                st.success("Thank you for your feedback! Your rating has been saved to the database.")
            except Exception as e:
                st.error(f"Error saving feedback: {str(e)}")

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
