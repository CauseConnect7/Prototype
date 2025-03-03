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

# 加载环境变量
load_dotenv()

# 设置OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

# ----------- 函数定义 -----------
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

def find_top_100_matches(embedding, looking_for_type):
    """Find top 100 matching organizations based on embedding similarity."""
    matches = []
    collection = collection1 if looking_for_type == "Non Profit" else collection2
    
    try:
        st.write(f"Searching in {looking_for_type} database...")
        for org in collection.find({"Embedding": {"$exists": True}}):
            if org.get("Embedding"):
                # 从BSON Binary转换回numpy数组
                org_embedding = np.frombuffer(org["Embedding"], dtype=np.float32)
                # 计算相似度
                similarity = 1 - cosine(embedding, org_embedding)
                matches.append((
                    similarity,
                    org.get("Name", "Unknown"),
                    org.get("Description", "No description available"),
                    org.get("Website", "N/A")
                ))
        
        matches.sort(reverse=True)
        st.write(f"Found {len(matches)} potential matches")
        return matches[:100]
    except Exception as e:
        st.error(f"Error finding matches: {str(e)}")
        return []

def scrape_company_mission(url):
    """Scrape the company's mission."""
    api_url = f"https://{os.getenv('RAPIDAPI_HOST')}/api/v1/scrape/content"
    headers = {
        "X-Rapidapi-Key": os.getenv("RAPIDAPI_KEY"),
        "X-Rapidapi-Host": os.getenv("RAPIDAPI_HOST"),
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": "What is the company's mission?",
        "url": url
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error scraping content from {url}")
        return None

# ----------- MongoDB连接 -----------
uri = os.getenv("MONGODB_URI")
client = MongoClient(uri)

# 连接到输入数据库
input_db = client[os.getenv("MONGODB_DB_INPUT_NAME")]  # Organization
# 连接到输出数据库
output_db = client[os.getenv("MONGODB_DB_OUTPUT_NAME")]  # UserInput

# 获取Profile集合
profile_collection = output_db[os.getenv("MONGODB_COLLECTION_PROFILE")]  # Profile

# ----------- Streamlit UI代码 -----------
# 设置页面配置
st.set_page_config(page_title="Organization Matcher", layout="wide")

# 创建主标题和介绍
st.title("Organization Partnership Matcher")
st.markdown("""
This tool helps you find potential partnership organizations that align with your values and goals.
Please fill in the information about your organization below.
""")

# 第一部分：组织档案
st.header("Section 1: Build Your Organization Profile")

# 创建两列布局
col1, col2 = st.columns(2)

with col1:
    org_name = st.text_input("Organization Name*", key="name")
    org_type = st.selectbox(
        "Organization Type*",
        ["Non Profit", "For-Profit"],
        key="type"
    )
    org_description = st.text_area(
        "Organization Mission Statement*",
        help="What is your organization's mission?",
        key="description"
    )

with col2:
    org_category = st.text_area(
        "Core Values*",
        help="What are the top three core values your brand stands for?",
        key="category"
    )
    target_audience = st.text_area(
        "Target Audience*",
        help="Who does your company serve? What social causes does your customer base care about?",
        key="audience"
    )

# 组合所有描述性信息为一个完整的描述
combined_description = f"""Organization Mission:
{org_description}

Core Values:
{org_category}

Target Audience:
{target_audience}"""

# 位置信息
col3, col4 = st.columns(2)
with col3:
    state = st.text_input("State", key="state")
with col4:
    city = st.text_input("City", key="city")

website = st.text_input("Website URL", key="website")

# 第二部分：匹配过程
st.header("Section 2: Start the Matching Process")

looking_for_type = st.selectbox(
    "Organization Type Looking For*",
    ["Non Profit", "For-Profit"],
    key="looking_for_type"
)

looking_for_description = st.text_area(
    "What Kind of Organization Are You Looking For?*",
    help="E.g.: A nonprofit that supports financial literacy for young adults, as our company provides budgeting and investment tools designed for first-time investors.",
    key="looking_for_desc"
)

# 提交按钮
if st.button("Find Matching Organizations"):
    if not all([org_name, org_type, org_description, org_category, target_audience, looking_for_type, looking_for_description]):
        st.error("Please fill in all required fields marked with *")
    else:
        row = {
            "Name": org_name,
            "Type": org_type,
            "Description": combined_description,
            "State": state,
            "City": city,
            "Website": website,
            "Organization looking 1": looking_for_type,
            "Organization looking 2": looking_for_description,
            "Target Audience": target_audience
        }
        
        # 尝试将用户输入的信息插入到Profile集合中
        try:
            profile_collection.insert_one(row)  # 直接插入到Profile集合
            st.success("User input successfully saved to MongoDB.")
        except Exception as e:
            st.error(f"Error saving user input to MongoDB: {str(e)}")

        with st.spinner("Finding matching organizations..."):
            generated_orgs = generate_ideal_organization(pd.Series(row))
            st.subheader("Suggested Organizations:")
            st.write(generated_orgs)
            
            tags = generate_fixed_tags(generated_orgs, row["Target Audience"])
            st.subheader("Generated Tags:")
            st.write(tags)
            
            if tags:
                embedding = get_embedding(tags)
                if embedding is not None:  # 直接检查embedding是否存在
                    matches = find_top_100_matches(embedding, looking_for_type)
                    
                    st.subheader("Top Matching Organizations:")
                    for i, match in enumerate(matches[:20], 1):
                        with st.expander(f"{i}. {match[1]}"):
                            st.write(f"Description: {match[2]}")
                            st.write(f"Website: {match[3]}")
                            if match[3] != "N/A":
                                mission_info = scrape_company_mission(match[3])
                                if mission_info:
                                    st.write("Mission Information:", mission_info)
