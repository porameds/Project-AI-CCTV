import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import schedule
import time

# ตั้งค่า DB
DB_CONFIG = {
    "user": "myuser",
    "password": "postgres",
    "host": "localhost",
    "port": "5432",
    "database": "computervision",
    "schema": "smart_ai"
}

# สร้าง connection string
conn_str = (
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)
engine = create_engine(conn_str)

def fetch_from_db():
    
    sql_query = """
        SELECT
            b.predict_time,
            b.detection_result,
            b.main_signal,
            b.sub_signal,
            CASE
                WHEN b.has_valid_signal THEN 'P'
                ELSE 'F'
            END AS fai_judge,
            b.avg_conf,
            b.machine_name,
            b.file_name ,
            b.group_time AS clean_date
        FROM (
            SELECT 
                file_name,
                MAX(predict_time) AS predict_time,
                MAX(detection_result) AS detection_result,
                MAX(main_signal) AS main_signal,
                MAX(sub_signal) AS sub_signal,
                MAX(machine_name) AS machine_name,
                MAX(avg_conf) AS avg_conf,
                DATE_TRUNC('hour', predict_time) AS group_time,
                MAX(
                    CASE 
                        WHEN main_signal = 1 AND sub_signal = 2 THEN 1 
                        ELSE 0
                    END
                ) = 1 AS has_valid_signal
            FROM smart_ai.oven_machine_test
            WHERE 
                main_signal = 1
                AND avg_conf > 0.5
                AND predict_time IS NOT NULL
            GROUP BY 
                file_name, main_signal, sub_signal, DATE_TRUNC('hour', predict_time)
        ) b
        WHERE b.group_time > NOW() - INTERVAL '7 days'
        ORDER BY b.group_time DESC;
    """
    df = pd.read_sql(sql_query, engine)
    return df

def insert_on_conflict(df, target_table="oven_summary_test"):
    """แทรกข้อมูลลงตาราง พร้อม ON CONFLICT DO NOTHING"""
    if df.empty:
        print(f"[{datetime.now()}] ไม่มีข้อมูลใหม่ให้เขียน ⛔")
        return

    temp_table = f"{target_table}_staging"
    schema = DB_CONFIG["schema"]

    # สร้าง staging table
    df.to_sql(temp_table, engine, schema=schema, index=False, if_exists="replace")
    print(f"[{datetime.now()}] เขียน staging table {schema}.{temp_table} แล้ว ✅")

    # SQL สำหรับ INSERT ... ON CONFLICT DO NOTHING
    insert_sql = f"""
        INSERT INTO {schema}.{target_table} (
            predict_time, detection_result, main_signal, sub_signal,
            fai_judge, avg_conf, machine_name, file_name
        )
        SELECT
            predict_time, detection_result, main_signal, sub_signal,
            fai_judge, avg_conf, machine_name, file_name
        FROM {schema}.{temp_table}
        ON CONFLICT (predict_time, file_name) DO NOTHING;
    """

    # Execute insert
    with engine.begin() as conn:
        conn.execute(text(insert_sql))

    print(f"[{datetime.now()}] เขียนข้อมูลลง {schema}.{target_table} สำเร็จ ✅ จำนวน {df.shape[0]} แถว")

def run_job():
    try:
        print(f"[{datetime.now()}] เริ่มทำงาน fetch + insert ...")
        df = fetch_from_db()
        insert_on_conflict(df)
    except Exception as e:
        print(f"[{datetime.now()}] ❌ ERROR: {e}")

# ตั้งเวลาให้ทำงานทุก 1 นาที
schedule.every(1).minutes.do(run_job)

if __name__ == "__main__":
    run_job()  # เรียกทันที
    while True:
        schedule.run_pending()
        time.sleep(10)
