"""
Fire Detection Dashboard
Real-time visualization of fire segmentation results
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from sqlalchemy import create_engine

# Page config
st.set_page_config(
    page_title="🔥 Fire Detection Dashboard",
    page_icon="🔥",
    layout="wide"
)

# Database connection string
DB_URL = "postgresql://airflow:airflow@postgres:5432/airflow"

def get_db_engine():
    """Get SQLAlchemy engine (creates new connection each time)"""
    try:
        return create_engine(DB_URL, pool_pre_ping=True)
    except Exception as e:
        st.error(f"Database engine creation failed: {e}")
        return None

def fetch_detections(limit=100):
    """Fetch recent fire detections"""
    query = f"""
        SELECT 
            camera_id,
            frame_number,
            detection_time,
            fire_detected,
            ROUND(fire_percentage::numeric, 2) as fire_percentage,
            ROUND(confidence::numeric, 3) as confidence
        FROM fire_detections
        ORDER BY detection_time DESC
        LIMIT {limit}
    """
    try:
        engine = get_db_engine()
        if engine is None:
            return pd.DataFrame()
        conn = engine.raw_connection()
        try:
            df = pd.read_sql(query, conn)
            if not df.empty:
                df['detection_time'] = pd.to_datetime(df['detection_time'])
            return df
        finally:
            conn.close()
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()

def fetch_statistics():
    """Fetch aggregated statistics"""
    query = """
        SELECT 
            COUNT(*) as total_frames,
            SUM(CASE WHEN fire_detected THEN 1 ELSE 0 END) as fire_frames,
            AVG(fire_percentage) as avg_fire_percentage,
            MAX(fire_percentage) as max_fire_percentage,
            AVG(confidence) as avg_confidence,
            camera_id
        FROM fire_detections
        GROUP BY camera_id
    """
    try:
        engine = get_db_engine()
        if engine is None:
            return pd.DataFrame()
        conn = engine.raw_connection()
        try:
            df = pd.read_sql(query, conn)
            return df
        finally:
            conn.close()
    except Exception as e:
        st.error(f"Statistics query failed: {e}")
        return pd.DataFrame()

def fetch_timeline_data(hours=1):
    """Fetch fire detection timeline"""
    query = f"""
        SELECT 
            DATE_TRUNC('minute', detection_time) as time_bucket,
            COUNT(*) as frame_count,
            AVG(fire_percentage) as avg_fire_pct,
            SUM(CASE WHEN fire_detected THEN 1 ELSE 0 END) as fire_count,
            camera_id
        FROM fire_detections
        WHERE detection_time >= NOW() - INTERVAL '{hours} hours'
        GROUP BY time_bucket, camera_id
        ORDER BY time_bucket DESC
    """
    try:
        engine = get_db_engine()
        if engine is None:
            return pd.DataFrame()
        conn = engine.raw_connection()
        try:
            df = pd.read_sql(query, conn)
            if not df.empty:
                df['time_bucket'] = pd.to_datetime(df['time_bucket'])
            return df
        finally:
            conn.close()
    except Exception as e:
        st.error(f"Timeline query failed: {e}")
        return pd.DataFrame()

def fetch_fire_images(limit=5):
    """Fetch recent fire detection images with masks"""
    query = f"""
        SELECT 
            camera_id,
            frame_number,
            detection_time,
            fire_percentage,
            image_base64,
            LENGTH(image_base64) as img_len
        FROM fire_detections
        WHERE fire_detected = true
        AND image_base64 IS NOT NULL
        ORDER BY detection_time DESC
        LIMIT {limit}
    """
    try:
        engine = get_db_engine()
        if engine is None:
            st.error("[DEBUG] Database engine is None")
            return pd.DataFrame()
        conn = engine.raw_connection()
        try:
            df = pd.read_sql(query, conn)
            st.write(f"[DEBUG] Query returned {len(df)} rows with images")
            if not df.empty:
                df['detection_time'] = pd.to_datetime(df['detection_time'])
                st.write(f"[DEBUG] Sample image length: {df.iloc[0]['img_len']} bytes")
            return df
        finally:
            conn.close()
    except Exception as e:
        st.error(f"Image query failed: {e}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

# Main app
def main():
    st.title("🔥 Fire Detection Dashboard")
    st.markdown("Real-time fire segmentation monitoring with U-Net")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)
        data_limit = st.slider("Recent Frames to Display", 50, 500, 100)
        timeline_hours = st.slider("Timeline Window (hours)", 1, 24, 1)
        
        st.divider()
        st.markdown("### 📊 About")
        st.info("""
        This dashboard displays real-time fire detection results using a U-Net segmentation model.
        
        **Metrics:**
        - Fire Percentage: % of frame containing fire
        - Confidence: Model prediction confidence
        - Detection Status: Fire detected if >1% of frame
        """)
        
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    # Fetch data
    with st.spinner("Loading data..."):
        df = fetch_detections(limit=data_limit)
        stats_df = fetch_statistics()
        timeline_df = fetch_timeline_data(hours=timeline_hours)
    
    if df.empty:
        st.warning("⚠️ No detection data available yet. The pipeline may still be starting.")
        st.info("💡 Check back in a few minutes or trigger the `fire_detection_lifecycle` DAG in Airflow.")
        st.stop()
    
    # Camera statistics
    if not stats_df.empty:
        st.header("📹 Camera Statistics")
        
        for _, row in stats_df.iterrows():
            with st.expander(f"📷 {row['camera_id']}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Frames Processed", f"{int(row['total_frames']):,}")
                with col2:
                    fire_rate = (row['fire_frames'] / row['total_frames'] * 100) if row['total_frames'] > 0 else 0
                    st.metric("🔥 Fire Rate", f"{fire_rate:.1f}%")
    
    st.divider()
    
    # Fire detection images
    st.header("🔥 Fire Detection Examples")
    st.markdown("Showing up to 5 most recent fire detections with mask overlays")
    
    fire_images_df = fetch_fire_images(limit=5)
    
    if not fire_images_df.empty:
        # Display images in columns
        num_images = len(fire_images_df)
        cols = st.columns(min(num_images, 3))
        
        for idx, row in fire_images_df.iterrows():
            col_idx = idx % 3
            with cols[col_idx]:
                st.markdown(f"**Camera {row['camera_id']} - Frame {row['frame_number']}**")
                st.markdown(f"🔥 Fire: {row['fire_percentage']:.1f}%")
                st.markdown(f"⏰ {row['detection_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Display image
                if pd.notna(row['image_base64']):
                    import base64
                    st.image(f"data:image/jpeg;base64,{row['image_base64']}", 
                            use_column_width=True,
                            caption=f"Fire detected: {row['fire_percentage']:.1f}%")
                st.divider()
    else:
        st.info("No fire detection images available yet. Images are saved when fire is detected.")
    
    # Recent detections table
    st.header("🔍 Recent Detections")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        show_fire_only = st.checkbox("Show only fire detections", value=False)
    with col2:
        min_fire_pct = st.slider("Minimum fire percentage", 0.0, 100.0, 0.0)
    
    # Filter data
    display_df = df.copy()
    if show_fire_only:
        display_df = display_df[display_df['fire_detected'] == True]
    display_df = display_df[display_df['fire_percentage'] >= min_fire_pct]
    
    # Style the dataframe
    def highlight_fire(row):
        if row['fire_detected']:
            return ['background-color: #ffcccc'] * len(row)
        return [''] * len(row)
    
    styled_df = display_df.style.apply(highlight_fire, axis=1).format({
        'fire_percentage': '{:.2f}%',
        'confidence': '{:.3f}'
    })
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Export data
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"fire_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.info(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Auto-refresh
    time.sleep(refresh_interval)
    st.rerun()

if __name__ == "__main__":
    main()
