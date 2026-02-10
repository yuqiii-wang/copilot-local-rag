from typing import List, Any, Optional
import datetime
import uuid
from pydantic import BaseModel
from data_manager import data_manager

# Load history into memory on module import (or first use)
MEMORY_CACHE = []
try:
    MEMORY_CACHE = data_manager.load_recent_records(7)
    print(f"Initial load: {len(MEMORY_CACHE)} records from last 7 days.")
except Exception as e:
    print(f"Failed to load history: {e}")

# Define the JSON Schema using Pydantic models
class ConversationTurn(BaseModel):
    ai_assistant: str
    human: str

class RefDoc(BaseModel):
    title: str
    source: str
    type: str
    score: Optional[float] = None
    comment: Optional[str] = None
    keywords: Optional[List[str]] = None

class QueryData(BaseModel):
    id: Optional[str] = None
    timestamp: Optional[str] = None 
    question: Optional[str] = None
    ref_docs: List[RefDoc] = []
    conversations: List[ConversationTurn]
    status: Optional[str] = None # 'accepted', 'rejected', or 'added_confluence'

class FrontendDataSchema(BaseModel):
    query: QueryData

class FeedbackSchema(BaseModel):
    id: str
    status: Optional[str] = None
    conversations: Optional[List[ConversationTurn]] = None

def save_memory_to_disk_today():
    """
    Backwards-compatible helper: saves ONLY records from today to the daily file.
    """
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    records_to_save = [r for r in MEMORY_CACHE if r.get('query', {}).get('timestamp', '').startswith(today_str)]
    if records_to_save:
        data_manager.overwrite_date(today_str, records_to_save)


def save_memory_to_disk_by_date():
    """
    Save all records in MEMORY_CACHE back to their respective daily files based on the
    date in the record's query.timestamp (YYYY-MM-DD). Records without timestamps
    will be saved to today's file.
    """
    # Group records by date
    per_date = {}
    for r in MEMORY_CACHE:
        ts = r.get('query', {}).get('timestamp', '') or ''
        if ts and len(ts) >= 10:
            date_key = ts[:10]
        else:
            date_key = datetime.datetime.now().strftime('%Y-%m-%d')
        per_date.setdefault(date_key, []).append(r)

    # Write each date file
    for date_key, recs in per_date.items():
        data_manager.overwrite_date(date_key, recs)

def _convert_to_file_url(path: str) -> str:
    """Helper: Converts local Windows paths to file:/// URLs for consistency."""
    if not path:
        return path
    s = str(path).strip()
    if s.lower().startswith("file:"):
        return s
    # Windows path detection (Drive letter or backslash)
    if ":" in s or "\\" in s:
        clean_path = s.replace('\\', '/')
        return f"file:///{clean_path}"
    return s

def process_initial_docs(raw_data: dict):
    """
    Step 1: Save the initial query, docs, and conversation.
    Returns the generated ID.
    """
    try:
        # Pre-process: Convert Windows paths in sources to file:/// URLs
        if 'query' in raw_data and 'ref_docs' in raw_data['query']:
            for doc in raw_data['query']['ref_docs']:
                if 'source' in doc:
                    doc['source'] = _convert_to_file_url(doc['source'])

        # Validate and parse input
        data = FrontendDataSchema(**raw_data)
        
        # Exclude the 1st conversation turn (backwards-compatible)
        if data.query.conversations and len(data.query.conversations) > 0:
            data.query.conversations.pop(0)
        # Remove any conversation turn where either side is empty/whitespace
        if data.query.conversations:
            data.query.conversations = [c for c in data.query.conversations if (c.ai_assistant and c.ai_assistant.strip()) and (c.human and c.human.strip())]

        # Generate ID and Timestamp if missing
        if not data.query.id:
            data.query.id = str(uuid.uuid4())
        if not data.query.timestamp:
            data.query.timestamp = datetime.datetime.now().isoformat()
            
        # Default status
        if not data.query.status:
            data.query.status = "pending"

        record_payload = data.model_dump()
        
        # Update memory
        MEMORY_CACHE.append(record_payload)
        
        # Persist
        save_memory_to_disk_today()
        
        return data.query.id
    except Exception as e:
        print(f"Error processing initial docs: {e}")
        return None

def process_feedback_update(feedback_data: dict):
    """
    Step 2: Update the status of an existing record.
    """
    try:
        fb = FeedbackSchema(**feedback_data)
        
        # Find record in memory
        found = False
        for record in MEMORY_CACHE:
            q = record.get("query", {})
            if q.get("id") == fb.id:
                # Update status only if provided
                if fb.status is not None:
                    q["status"] = fb.status

                # Append conversations if provided
                if fb.conversations:
                    new_convs = [c.model_dump() for c in fb.conversations]
                    # Filter out turns where either side is empty/dummy (require both non-empty)
                    valid_new_convs = [c for c in new_convs if (c.get('human') and str(c.get('human')).strip()) and (c.get('ai_assistant') and str(c.get('ai_assistant')).strip())]

                    if valid_new_convs:
                        # Append to existing
                        if "conversations" not in q:
                            q["conversations"] = []
                        q["conversations"].extend(valid_new_convs)
                found = True
                break
        
        if found:
            save_memory_to_disk_by_date()
            return True
        else:
            print(f"Record with ID {fb.id} not found in memory cache.")
            return False
            
    except Exception as e:
        print(f"Error processing feedback: {e}")
        return False


def process_update_docs(raw_data: dict):
    """
    Update existing record's ref_docs (merge comments/keywords).
    Accepts partial payloads like { "query": { "id": "...", "ref_docs": [...] } }
    """
    try:
        q = raw_data.get('query', {}) or {}
        qid = q.get('id')
        if not qid:
            print("No query id provided for update_docs")
            return False

        new_refs = q.get('ref_docs', []) or []
        if not isinstance(new_refs, list):
            print("ref_docs must be a list")
            return False

        # Pre-process: Convert Windows paths to file:/// URLs
        for nr in new_refs:
            if 'source' in nr:
                nr['source'] = _convert_to_file_url(nr['source'])

        found = False
        for record in MEMORY_CACHE:
            existing_q = record.get('query', {})
            if existing_q.get('id') == qid:
                # Merge ref_docs by normalized source
                existing_docs = existing_q.get('ref_docs', [])

                def normalize(src: str):
                    s = (src or '').lower()
                    if s.startswith('file:///'):
                        s = s[8:]
                    elif s.startswith('file://'):
                        s = s[7:]
                    return s.replace('\\', '/').strip()

                existing_map = {normalize(d.get('source','')): d for d in existing_docs}

                for new_ref in new_refs:
                    nsrc = normalize(new_ref.get('source',''))
                    if nsrc in existing_map:
                        # update fields like comment and keywords
                        dest = existing_map[nsrc]
                        if new_ref.get('comment'):
                            dest['comment'] = new_ref.get('comment')
                        if new_ref.get('keywords'):
                            dest['keywords'] = new_ref.get('keywords')
                        if 'score' in new_ref and new_ref['score'] is not None:
                            dest['score'] = new_ref['score']
                    else:
                        # append new ref
                        existing_docs.append(new_ref)

                # Persist change
                existing_q['ref_docs'] = existing_docs
                save_memory_to_disk_by_date()
                found = True
                break

        return found
    except Exception as e:
        print(f"Error in process_update_docs: {e}")
        return False


def process_delete_doc(qid: str, source: str):
    """Delete a reference document from a record identified by qid and source."""
    try:
        if not qid or not source:
            return False

        def normalize(src: str):
            s = (src or '').lower()
            if s.startswith('file:///'):
                s = s[8:]
            elif s.startswith('file://'):
                s = s[7:]
            return s.replace('\\', '/').strip()

        nsrc = normalize(source)
        found = False
        for record in MEMORY_CACHE:
            q = record.get('query', {})
            if q.get('id') == qid:
                docs = q.get('ref_docs', [])
                new_docs = [d for d in docs if normalize(d.get('source','')) != nsrc]
                if len(new_docs) != len(docs):
                    q['ref_docs'] = new_docs
                    save_memory_to_disk_by_date()
                    found = True
                break
        return found
    except Exception as e:
        print(f"Error in process_delete_doc: {e}")
        return False


def process_delete_query(qid: str):
    """Delete an entire query record by id."""
    try:
        if not qid:
            return False
        initial_len = len(MEMORY_CACHE)
        MEMORY_CACHE[:] = [r for r in MEMORY_CACHE if r.get('query', {}).get('id') != qid]
        if len(MEMORY_CACHE) != initial_len:
            save_memory_to_disk_by_date()
            return True
        return False
    except Exception as e:
        print(f"Error in process_delete_query: {e}")
        return False

# Preserved for backward compatibility if needed, but we should switch routes to use above
def process_and_save_frontend_data(raw_data: dict):
    return process_initial_docs(raw_data) is not None

