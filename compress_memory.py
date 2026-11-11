# compress_memory.py
import os
import sqlite3
import ollama
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

DB_PATH = os.getenv("DB_PATH", "/home/dhirajpatra/sonku_memory.db")
MODEL_NAME = os.getenv("SLM_MODEL", "qwen3.5:4b")
COMPRESSION_THRESHOLD_MB = int(os.getenv("COMPRESSION_THRESHOLD_MB", "50"))

SIZE_THRESHOLD_BYTES = COMPRESSION_THRESHOLD_MB * 1024 * 1024

def get_file_size(filepath):
    if os.path.exists(filepath):
        return os.path.getsize(filepath)
    return 0

def compress_memory():
    current_size = get_file_size(DB_PATH)
    print(f"[Sonku Memory Maintenance] Current DB Size: {current_size / (1024 * 1024):.2f} MB")

    if current_size < SIZE_THRESHOLD_BYTES:
        print(f"[Sonku Memory Maintenance] DB size is below threshold ({COMPRESSION_THRESHOLD_MB} MB). Skipping compression.")
        return

    print("[Sonku Memory Maintenance] Size threshold exceeded! Starting LLM summarization & pruning...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT person_id FROM interactions")
    people = [row[0] for row in cursor.fetchall()]

    for person_id in people:
        cursor.execute("""
            SELECT id, user_text, robot_text, visual_context 
            FROM interactions 
            WHERE person_id = ? 
            ORDER BY id ASC 
            LIMIT -1 OFFSET 10
        """, (person_id,))
        old_rows = cursor.fetchall()

        if not old_rows:
            continue

        cursor.execute("SELECT summary_memory FROM person_profiles WHERE person_id = ?", (person_id,))
        res = cursor.fetchone()
        existing_summary = res[0] if res else "New acquaintance."

        history_text = "\n".join([f"User: {r[1]} | Sonku: {r[2]} | Context: {r[3]}" for r in old_rows])
        prompt = f"""Existing Profile Summary for {person_id}:
{existing_summary}

Older Raw Interaction History:
{history_text}

Task: Compress the key facts, preferences, user traits, and important details into a single dense summary paragraph (max 4 sentences). Keep critical memory intact."""

        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
        new_summary = response['message']['content'].strip()

        cursor.execute("""
            INSERT OR REPLACE INTO person_profiles (person_id, summary_memory, last_seen)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (person_id, new_summary))

        row_ids_to_delete = [r[0] for r in old_rows]
        cursor.executemany("DELETE FROM interactions WHERE id = ?", [(rid,) for rid in row_ids_to_delete])
        conn.commit()

        print(f"[Sonku Memory Maintenance] Compressed {len(row_ids_to_delete)} rows for user '{person_id}'.")

    print("[Sonku Memory Maintenance] Running SQLite VACUUM to shrink disk allocation...")
    cursor.execute("VACUUM;")
    conn.close()

    new_size = get_file_size(DB_PATH)
    print(f"[Sonku Memory Maintenance] Compression complete. New DB Size: {new_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    compress_memory()