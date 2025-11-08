"""
SE363 - Big Data Exercise 1
Streamlit Dashboard - Đơn giản và nhanh
"""

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Cấu hình trang
st.set_page_config(
    page_title="Đếm Xe Real-Time",
    page_icon="🚗",
    layout="wide"
)

# Cấu hình database
DB_CONFIG = {
    "user": "airflow",
    "password": "airflow",
    "host": "postgres",
    "port": 5432,
    "database": "airflow"
}

@st.cache_data(ttl=5)
def load_simple_stats():
    """Chỉ lấy stats đơn giản - nhanh hơn"""
    engine = create_engine(
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    
    query = """
        SELECT 
            camera_id,
            vehicle_type,
            SUM(count) as total_count
        FROM vehicle_counts
        GROUP BY camera_id, vehicle_type
        ORDER BY total_count DESC
        LIMIT 50
    """
    
    try:
        conn = engine.raw_connection()
        try:
            df = pd.read_sql(query, conn)
        finally:
            conn.close()
        return df
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=5)
def load_recent_data():
    """Lấy 10 dòng dữ liệu gần nhất"""
    engine = create_engine(
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    
    query = """
        SELECT 
            camera_id,
            vehicle_type,
            count,
            confidence,
            detection_time
        FROM vehicle_counts
        ORDER BY frame_number DESC
        LIMIT 10
    """
    
    try:
        conn = engine.raw_connection()
        try:
            df = pd.read_sql(query, conn)
        finally:
            conn.close()
        return df
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return pd.DataFrame()

# Header
st.title("🚗 Hệ Thống Đếm Xe Real-Time")
st.markdown("**Đơn giản - Nhanh - Hiệu quả**")

# Auto-refresh mỗi 10 giây
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=10 * 1000, limit=None, key="auto_refresh")

if st.sidebar.button("🔄 Làm mới ngay"):
    st.rerun()

# Load data
stats_df = load_simple_stats()
recent_df = load_recent_data()

# Metrics tổng quan
st.markdown("### 📊 Tổng Quan")

if not stats_df.empty:
    col1, col2, col3 = st.columns(3)
    
    total = stats_df['total_count'].sum()
    cameras = stats_df['camera_id'].nunique()
    types = stats_df['vehicle_type'].nunique()
    
    col1.metric("🚗 Tổng số xe", f"{int(total)}")
    col2.metric("📹 Camera hoạt động", f"{cameras}")
    col3.metric("🔢 Loại xe phát hiện", f"{types}")
else:
    st.info("⏳ Chưa có dữ liệu...")

st.markdown("---")

# Đếm theo loại xe
st.markdown("### � Đếm Theo Loại Xe")

if not stats_df.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Camera 1**")
        cam1_df = stats_df[stats_df['camera_id'] == 'camera1']
        if not cam1_df.empty:
            for _, row in cam1_df.iterrows():
                st.write(f"- **{row['vehicle_type'].upper()}**: {int(row['total_count'])} xe")
        else:
            st.write("_Chưa có dữ liệu_")
    
    with col2:
        st.markdown("**Camera 2**")
        cam2_df = stats_df[stats_df['camera_id'] == 'camera2']
        if not cam2_df.empty:
            for _, row in cam2_df.iterrows():
                st.write(f"- **{row['vehicle_type'].upper()}**: {int(row['total_count'])} xe")
        else:
            st.write("_Chưa có dữ liệu_")
else:
    st.info("Chưa có dữ liệu thống kê")

st.markdown("---")

# Dữ liệu gần nhất
st.markdown("### 🕐 10 Phát Hiện Gần Nhất")

if not recent_df.empty:
    # Format dữ liệu
    display_df = recent_df.copy()
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.0%}")
    display_df.columns = ['Camera', 'Loại xe', 'Số lượng', 'Độ chính xác', 'Thời gian']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("Chưa có dữ liệu phát hiện")

# Footer
st.markdown("---")
current_time = datetime.now().strftime('%H:%M:%S')
st.caption(f"⏰ Cập nhật lúc: {current_time} | 🔄 Tự động làm mới mỗi 10 giây")
