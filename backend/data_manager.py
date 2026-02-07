import os
import logging
import json
import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        # Using backend/data relative to this file (backend/data_manager.py)
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.records_dir = os.path.join(self.data_dir, "records")
        self.imgs_dir = os.path.join(self.data_dir, "imgs")
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.records_dir):
            os.makedirs(self.records_dir)
        if not os.path.exists(self.imgs_dir):
            os.makedirs(self.imgs_dir)

        self._initialized = True

    def connect(self):
        """No-op for offline mode"""
        logger.info(f"Using offline mode. Data will be recorded locally in {self.data_dir}")

    def disconnect(self):
        """No-op for offline mode"""
        pass

    def get_connection(self):
        """
        Dummy context manager for compatibility with existing code.
        """
        class DummyConnection:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc_val, exc_tb): pass
            def cursor(self): return self
            def execute(self, query, params=None): pass
            def fetchall(self): return []
            def commit(self): pass
            def rollback(self): pass
            
        return DummyConnection()

    def _record_offline(self, query, params, fetch):
        """
        Record the query execution to a local JSON file.
        Stores as a list of objects in the JSON file.
        """
        try:
            # Process params to save binary data to local imgs folder
            processed_params = []
            if params:
                for param in params:
                    if isinstance(param, bytes):
                        img_filename = f"offline_{uuid.uuid4()}.bin"
                        img_path = os.path.join(self.imgs_dir, img_filename)
                        try:
                            with open(img_path, "wb") as f:
                                f.write(param)
                            processed_params.append(f"saved_file::{img_filename}")
                        except Exception as e:
                            logger.error(f"Failed to write image file: {e}")
                            processed_params.append("<binary_error>")
                    else:
                        processed_params.append(param)
            else:
                processed_params = params

            # Helper for JSON serialization
            def json_serial(obj):
                if isinstance(obj, (datetime.datetime, datetime.date)):
                    return obj.isoformat()
                if isinstance(obj, bytes):
                    return "<binary_omitted>"
                return str(obj)

            # Construct the record
            # If this is a specialized store command, unwrap the payload for cleaner JSON
            if query == "STORE_FRONTEND_DATA" and isinstance(processed_params, list) and len(processed_params) > 0:
                 record = processed_params[0]
                 # check if record is unwrapped
                 if isinstance(record, dict) and 'query' in record and 'id' in record['query']:
                     # It's already in the structure we want, so use it directly
                     pass
                 else:
                     # It's a raw payload, wrap if needed or just use
                     if isinstance(record, dict):
                        if "id" not in record:
                            record["id"] = str(uuid.uuid4())
                        if "timestamp" not in record:
                            record["timestamp"] = datetime.datetime.now().isoformat()
            else:
                record = {
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "query": query,
                    "params": processed_params,
                    "fetch": fetch
                }

            # Daily rotation: filename includes date
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            filename = os.path.join(self.records_dir, f"data_{date_str}.json")

            # Read existing list or create new
            current_data = []
            if os.path.exists(filename):
                try:
                    with open(filename, "r", encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            try:
                                # Try parsing as standard JSON list
                                loaded = json.loads(content)
                                if isinstance(loaded, list):
                                    current_data = loaded
                                elif isinstance(loaded, dict):
                                    current_data = [loaded]
                            except json.JSONDecodeError:
                                # Fallback: Try parsing as JSON Lines (legacy format)
                                lines = content.strip().split('\n')
                                for line in lines:
                                    if line.strip():
                                        try:
                                            current_data.append(json.loads(line))
                                        except: pass
                except Exception as e:
                    logger.warning(f"Error reading existing file {filename}: {e}. Starting fresh.")
                    pass

            # Append new record
            current_data.append(record)

            # Write back as detailed JSON list (rewriting file)
            with open(filename, "w", encoding='utf-8') as f:
                f.write(json.dumps(current_data, default=json_serial, indent=2))
            
            logger.info(f"Recorded query to offline file: {filename}")

            if fetch:
                # return a dummy result structure ID, created_at
                dummy_id = int(str(uuid.uuid4().int)[:8]) # small int
                return [[dummy_id, datetime.datetime.now()]]
            
            # Return the record ID so caller can track it
            if isinstance(record, dict) and "id" in record:
                return record["id"]

        except Exception as e:
            logger.error(f"Failed to record offline data: {e}")
        
        return None

    def overwrite_today(self, records):
        """
        Overwrite today's log file with the provided list of records.
        Useful for updates to existing records.
        """
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        return self.overwrite_date(date_str, records)

    def overwrite_date(self, date_str: str, records):
        """
        Overwrite the log file for a specific date (YYYY-MM-DD) with the provided list of records.
        """
        try:
            filename = os.path.join(self.records_dir, f"data_{date_str}.json")
            
            def json_serial(obj):
                if isinstance(obj, (datetime.datetime, datetime.date)):
                    return obj.isoformat()
                return str(obj)

            with open(filename, "w", encoding='utf-8') as f:
                f.write(json.dumps(records, default=json_serial, indent=2))
            
            logger.info(f"Overwrote offline file with {len(records)} records: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to overwrite offline data for {date_str}: {e}")
            return False

    def load_recent_records(self, days=7):
        """
        Load records from the last N days.
        """
        records = []
        today = datetime.datetime.now()
        for i in range(days):
            date_val = today - datetime.timedelta(days=i)
            date_str = date_val.strftime("%Y-%m-%d")
            filename = os.path.join(self.records_dir, f"data_{date_str}.json")
            
            if os.path.exists(filename):
                try:
                    with open(filename, "r", encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            try:
                                data = json.loads(content)
                                if isinstance(data, list):
                                    records.extend(data)
                                elif isinstance(data, dict):
                                    records.append(data)
                            except json.JSONDecodeError:
                                # Fallback for JSON Lines
                                lines = content.strip().split('\n')
                                for line in lines:
                                    if line.strip():
                                        try:
                                            records.append(json.loads(line))
                                        except: pass
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        
        return records

    def execute_query(self, query, params=None, fetch=False):
        """
        Log query to JSON file.
        """
        return self._record_offline(query, params, fetch)

# Create a global instance
data_manager = DataManager()
