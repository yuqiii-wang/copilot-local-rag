from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from typing import Optional, List
import os
from services import data_service
from config import config
from pydantic import BaseModel
import queue
import asyncio
import contextlib
import sys

router = APIRouter()

ADMIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'admin')
GRAPH_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'services', 'knowledge_graph', 'knowledge_graph.png'))

@router.get('/', response_class=HTMLResponse)
async def admin_index():
    index_path = os.path.join(ADMIN_DIR, 'index.html')
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail='Admin UI not found')
    return FileResponse(index_path)

@router.get('/records')
async def get_records():
    return JSONResponse(content={'records': data_service.MEMORY_CACHE})

@router.post('/update_doc')
async def update_doc(payload: dict):
    success = data_service.process_update_docs(payload)
    if not success:
        raise HTTPException(status_code=404, detail='Record not found or failed to update')
    return {'status': 'ok'}

class ManualDoc(BaseModel):
    url: str
    score: float

class ManualRecordRequest(BaseModel):
    question: str
    docs: List[ManualDoc]

@router.post('/manual_record')
async def add_manual_record(req: ManualRecordRequest):
    # Logic Consolidation:
    # 1. Calculate stats: total sum, which docs have 0.
    # 2. If total > 0:
    #    a. If total > 100: Normalize all down to 100.
    #    b. If total <= 100:
    #       i. If there are docs with 0 score: Distribute (100 - total) among them.
    #       ii. Otherwise: Keep scores as is.
    # 3. If total == 0 (no manual scores):
    #    a. Apply heuristic (60% rule).

    NORM_MAX = float(config.SCORE_NORMALIZATION_FACTOR)
    total_score = sum(d.score for d in req.docs)
    zero_indices = [i for i, d in enumerate(req.docs) if d.score == 0]
    
    # Pre-calculate fill value or normalization factor
    fill_value = 0.0
    normalize_factor = 1.0
    use_heuristic = False
    heuristic_scores = []
    norm_docs = []

    if total_score == 0:
        use_heuristic = True
        remaining = NORM_MAX
        count = len(req.docs)
        for i in range(count):
            if i == count - 1:
                heuristic_scores.append(remaining)
            else:
                val = remaining * 0.6
                heuristic_scores.append(val)
                remaining -= val
    else:
        # User provided some scores
        if total_score > NORM_MAX:
             normalize_factor = NORM_MAX / total_score
        elif total_score < NORM_MAX and zero_indices:
             remaining = NORM_MAX - total_score
             fill_value = remaining / len(zero_indices)

    for i, d in enumerate(req.docs):
        if use_heuristic:
            norm_score = heuristic_scores[i]
        elif total_score > NORM_MAX:
            norm_score = d.score * normalize_factor
        elif d.score == 0 and fill_value > 0:
            norm_score = fill_value
        else:
            norm_score = d.score
            
        # Determine type based on extension
        _, ext = os.path.splitext(d.url)
        ext = ext.lower()
        # Common code extensions (aligned with indexers + common web/backend langs)
        code_exts = {
            '.py', '.js', '.ts', '.tsx', '.jsx', 
            '.java', '.cpp', '.hpp', '.h', '.c', '.cc', 
            '.sh', '.bash', '.sql', '.go', '.rs', '.php', '.rb',
            '.css', '.scss', '.json', '.xml', '.yml', '.yaml'
        }
        
        doc_type = "code" if ext in code_exts else "confluence"
        
        # Determine title from URL (last segment)
        clean_url = d.url.replace('\\', '/')
        title_val = clean_url.split('/')[-1] if clean_url else "Untitled"

        norm_docs.append({
            "source": d.url,
            "title": title_val, 
            "type": doc_type,
            "score": norm_score,
            "comment": "",
            "keywords": []
        })

    payload = {
        "query": {
            "question": req.question,
            "conversations": [], # Required empty list
            "ref_docs": norm_docs,
            "status": "accepted" # Treat manual entries as accepted
        }
    }
    
    new_id = data_service.process_initial_docs(payload)
    if not new_id:
         raise HTTPException(status_code=500, detail="Failed to save record")
         
    return {"status": "ok", "id": new_id}

# Status flags for background jobs
training_running = False
visualize_running = False

# Thread-safe queue for training logs
training_log_queue: "queue.Queue[str]" = queue.Queue()

class QueueWriter:
    def __init__(self, q: "queue.Queue[str]"):
        self.q = q
    def write(self, s: str):
        # Push each line/fragment into queue (strip trailing newlines)
        for part in s.splitlines():
            if part is not None:
                self.q.put(str(part))
    def flush(self):
        pass

@router.post('/train')
async def trigger_train(background_tasks: BackgroundTasks, init_lr: Optional[float] = None):
    """Start training in background. Optional query param:
    - init_lr: float (initial learning rate to set in the training module)
    """
    global training_running
    if training_running:
        return {'status': 'already_running'}

    def _run_train(lr):
        global training_running
        try:
            training_running = True
            from services.knowledge_graph import train_model

            if lr is not None:
                try:
                    train_model.LEARNING_RATE = float(lr)
                    training_log_queue.put(f"Init LR set to {train_model.LEARNING_RATE}")
                except Exception as ex:
                    training_log_queue.put(f"Invalid init_lr provided: {lr} ({ex})")

            # Redirect stdout to queue writer so we can stream logs
            qw = QueueWriter(training_log_queue)
            with contextlib.redirect_stdout(qw):
                try:
                    train_model.train()
                except Exception as exc:
                    # ensure exception gets into queue
                    training_log_queue.put(f"Train error: {exc}")
                    raise
        except Exception as exc:
            print('Train error', exc)
        finally:
            training_running = False
            # Signal end
            training_log_queue.put("[TRAIN_DONE]")

    background_tasks.add_task(_run_train, init_lr)
    return {'status': 'started'}

@router.post('/visualize')
async def trigger_visualize(background_tasks: BackgroundTasks):
    global visualize_running
    if visualize_running:
        return {'status': 'already_running'}
    def _run_vis():
        global visualize_running
        try:
            visualize_running = True
            from services.knowledge_graph.visualize_graph import visualize_knowledge_graph
            dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'services', 'knowledge_graph', 'dummy_dataset')
            visualize_knowledge_graph(dataset_dir, output_file=os.path.basename(GRAPH_FILE), max_keywords=200, max_edges=400)
        except Exception as exc:
            print('Visualize error', exc)
        finally:
            visualize_running = False

    background_tasks.add_task(_run_vis)
    return {'status': 'started'}

@router.get('/status')
async def get_status():
    return {'training': training_running, 'visualize': visualize_running}

@router.get('/stats')
async def get_stats(range: str = '7d'):
    """Return time-series counts for recent records and top questions.
    range: '7d','30d','90d','365d','all' or days as integer
    Robust to invalid timestamps: skips bad entries and logs examples.
    """
    try:
        import datetime
        now = datetime.datetime.utcnow()
        if range == 'all':
            start = None
        else:
            try:
                if range.endswith('d'):
                    days = int(range[:-1])
                else:
                    days = int(range)
            except Exception:
                # fallback to 7 days
                days = 7
            start = now - datetime.timedelta(days=days)

        # Build date buckets: use daily buckets between start and now
        date_counts = {}
        questions = {}
        bad_ts_examples = []

        for rec in data_service.MEMORY_CACHE:
            q = rec.get('query', {})
            ts = q.get('timestamp')
            if not ts:
                continue

            dt = None
            try:
                if isinstance(ts, str):
                    # Try ISO formats first
                    try:
                        dt = datetime.datetime.fromisoformat(ts)
                    except Exception:
                        try:
                            dt = datetime.datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.%fZ')
                        except Exception:
                            try:
                                dt = datetime.datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S')
                            except Exception:
                                raise
                elif isinstance(ts, datetime.datetime):
                    dt = ts
                else:
                    # Unknown type
                    raise ValueError('Unsupported timestamp type')
            except Exception:
                # Record a small sample of bad timestamps for debugging and skip
                if len(bad_ts_examples) < 3:
                    bad_ts_examples.append(str(ts))
                continue

            # Normalize timezone-aware datetimes to naive UTC for safe comparisons
            try:
                if dt.tzinfo is not None:
                    dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
            except Exception:
                # If normalization fails, skip the timestamp
                if len(bad_ts_examples) < 3:
                    bad_ts_examples.append(str(ts))
                continue

            if start and dt < start:
                continue

            date_key = dt.date().isoformat()
            date_counts[date_key] = date_counts.get(date_key, 0) + 1
            qtext = (q.get('question') or '').strip()
            if qtext:
                questions[qtext] = questions.get(qtext, 0) + 1

        # Fill missing dates (if start is set)
        labels = []
        counts = []
        if start:
            cur = start.date()
            endd = now.date()
            while cur <= endd:
                k = cur.isoformat()
                labels.append(k)
                counts.append(date_counts.get(k, 0))
                cur = cur + datetime.timedelta(days=1)
        else:
            # sort keys
            keys = sorted(date_counts.keys())
            labels = keys
            counts = [date_counts[k] for k in keys]

        top_questions = sorted([{'question': k, 'count': v} for k, v in questions.items()], key=lambda x: x['count'], reverse=True)[:20]

        if bad_ts_examples:
            print(f"/admin/stats: skipped {len(bad_ts_examples)} records with invalid timestamps (examples: {bad_ts_examples})")

        return {'labels': labels, 'counts': counts, 'top_questions': top_questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/train_logs')
async def train_logs():
    """Stream training logs as Server-Sent Events (SSE)."""
    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            try:
                # Blocking get in threadpool to avoid blocking event loop
                line = await loop.run_in_executor(None, training_log_queue.get, True, 1.0)
            except Exception:
                # timeout
                if not training_running and training_log_queue.empty():
                    break
                continue
            if line == "[TRAIN_DONE]":
                yield f"data: {line}\n\n"
                break
            # Escape newlines already handled; send as single data event
            yield f"data: {line}\n\n"
        yield "data: [END]\n\n"
    return StreamingResponse(event_generator(), media_type='text/event-stream')
@router.get('/graph.png')
async def get_graph():
    if os.path.exists(GRAPH_FILE):
        return FileResponse(GRAPH_FILE, media_type='image/png')
    else:
        raise HTTPException(status_code=404, detail='Graph not found')

@router.post('/delete_doc')
async def delete_doc(payload: dict):
    # payload: { query: { id: '...', ref_docs: [{ source: '...'}] } }
    try:
        q = payload.get('query', {}) or {}
        qid = q.get('id')
        refs = q.get('ref_docs', []) or []
        if not qid or not refs:
            raise HTTPException(status_code=400, detail='Missing id or ref_docs')
        source = refs[0].get('source')
        success = data_service.process_delete_doc(qid, source)
        if not success:
            raise HTTPException(status_code=404, detail='Record or doc not found')
        return {'status': 'ok'}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/delete_query')
async def delete_query(payload: dict):
    # payload: { id: '...' }
    try:
        qid = payload.get('id')
        if not qid:
            raise HTTPException(status_code=400, detail='Missing id')
        success = data_service.process_delete_query(qid)
        if not success:
            raise HTTPException(status_code=404, detail='Record not found')
        return {'status': 'ok'}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
