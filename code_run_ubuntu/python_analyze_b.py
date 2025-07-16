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
        WITH ranked_data AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY file_name
                       ORDER BY predict_time DESC
                   ) AS row_num
            FROM smart_ai.oven_machine_b
            WHERE 
                main_signal IN (0, 1)
                AND avg_conf > -1
                AND predict_time IS NOT NULL
        )
        SELECT
            predict_time,
            detection_result,
            main_signal,
            sub_signal,
            CASE 
                WHEN main_signal = 1 AND sub_signal in (2,3) THEN 'P'
                WHEN detection_result = 'no detection'  THEN 'no detect'
                ELSE 'F'
            END AS fai_judge,
            avg_conf,
            machine_name,
            file_name,
            DATE_TRUNC('hour', predict_time) AS clean_date
        FROM ranked_data
        WHERE row_num = 1
          AND predict_time > NOW() - INTERVAL '7 days'
        ORDER BY predict_time DESC;
    """
    df = pd.read_sql(sql_query, engine)
    return df

def insert_on_conflict(df, target_table="oven_summary_test_b"):
    """แทรกข้อมูลลงตาราง พร้อม ON CONFLICT DO NOTHING"""
    if df.empty:
        print(f"[{datetime.now()}] ไม่มีข้อมูลใหม่ให้เขียน ")
        return
    
    temp_table = f"{target_table}_staging"
    schema = DB_CONFIG["schema"]
    # สร้าง staging table
    df.to_sql(temp_table, engine, schema=schema, index=False, if_exists="replace")
    print(f"[{datetime.now()}] เขียน staging table {schema}.{temp_table} แล้ว ")

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
    print(f"[{datetime.now()}] เขียนข้อมูลลง {schema}.{target_table} สำเร็จ จำนวน {df.shape[0]} แถว")

def count_fai_judge_summary():
    query = """
            SELECT
                DATE_TRUNC('hour', predict_time) AS predict_hour,
                SUM(CASE WHEN main_signal  = '1' THEN 1 ELSE 0 END) AS machine_open,
                SUM(CASE WHEN fai_judge = 'P' THEN 1 ELSE 0 END) AS pass_count,
                SUM(CASE WHEN fai_judge = 'F' THEN 1 ELSE 0 END) AS fail_count,
                SUM(CASE WHEN main_signal = '1' AND machine_name = 'machine-1' THEN 1 ELSE 0 END) AS oven_1,  
             	SUM(CASE WHEN main_signal = '1' AND machine_name = 'machine-2' THEN 1 ELSE 0 END) AS oven_2, 
                SUM(CASE WHEN main_signal = '1' AND machine_name = 'machine-3' THEN 1 ELSE 0 END) AS oven_3,
                SUM(CASE WHEN main_signal = '1' AND machine_name = 'machine-4' THEN 1 ELSE 0 END) AS oven_4,
                SUM(CASE WHEN fai_judge = 'no detect' THEN 1 ELSE 0 END) AS no_detect
            FROM smart_ai.oven_summary_test_b
            GROUP BY DATE_TRUNC('hour', predict_time)
            ORDER BY predict_hour DESC;
    """
    df_1 = pd.read_sql(query, engine)
    return df_1

def insert_on_conflict_fai_judge(df_1, target_table="oven_summary_count_fai_judge_b"):
    if df_1.empty:
        print(f"[{datetime.now()}] ไม่มีข้อมูลใหม่ให้เขียน ")
        return

    temp_table = f"{target_table}_staging"
    schema = DB_CONFIG["schema"]

    # ลบค่าที่ซ้ำกันใน staging ก่อน (กรณีซ้ำจาก pandas)
    df_1.drop_duplicates(subset=["predict_hour"], inplace=True)

    # สร้าง staging table
    df_1.to_sql(temp_table, engine, schema=schema, index=False, if_exists="replace")
    print(f"[{datetime.now()}] เขียน staging table {schema}.{temp_table} แล้ว ")

    # SQL ลบค่าที่มี predict_hour ซ้ำจากตารางจริง
    delete_sql = f"""
        DELETE FROM {schema}.{target_table} t
        USING {schema}.{temp_table} s
        WHERE t.predict_hour = s.predict_hour;
    """
    # SQL แทรกข้อมูลใหม่เข้าไป
    insert_sql = f"""
        INSERT INTO {schema}.{target_table} (
            predict_hour, machine_open, pass_count, fail_count, oven_1, oven_2, oven_3, oven_4, no_detect
        )
        SELECT
            predict_hour, machine_open, pass_count, fail_count, oven_1, oven_2, oven_3, oven_4, no_detect
        FROM {schema}.{temp_table};
    """
    # Execute delete + insert
    with engine.begin() as conn:
        conn.execute(text(delete_sql))
        conn.execute(text(insert_sql))
    print(f"[{datetime.now()}] ลบและแทรกข้อมูลใน {schema}.{target_table} สำเร็จ จำนวน {df_1.shape[0]} แถว")


def run_job():

    try:
        print(f"[{datetime.now()}] เริ่มทำงาน fetch + insert ...")
        df = fetch_from_db()
        insert_on_conflict(df)

        print(f"[{datetime.now()}] start fetch + insert fai count ...")
        df_1 = count_fai_judge_summary()
        insert_on_conflict_fai_judge(df_1)

        print(f"[{datetime.now()}] ✅ Schedule working")

    except Exception as e:
        print(f"[{datetime.now()}]  ERROR: {e}")
# ตั้งเวลาให้ทำงานทุก 1 นาที
schedule.every(1).minutes.do(run_job)

if __name__ == "__main__":
    run_job()  # เรียกทันที
    while True:
        schedule.run_pending()
        time.sleep(10)
