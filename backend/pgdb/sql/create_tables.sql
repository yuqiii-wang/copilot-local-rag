-- Active: 1769095296944@@127.0.0.1@5431@repo_ask
-- Switch to the application schema
SET search_path TO repo_ask;


-- table for storing OCR results
CREATE TABLE IF NOT EXISTS repo_ask.ocr_results (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    ocr_text TEXT,
    image BYTEA, -- Store image binary data
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- table for storing individual bounding boxes from OCR
CREATE TABLE IF NOT EXISTS repo_ask.bbox (
    id SERIAL PRIMARY KEY,
    ocr_result_id INTEGER REFERENCES repo_ask.ocr_results(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    confidence FLOAT,
    x1 FLOAT, -- [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    y1 FLOAT,
    x2 FLOAT,
    y2 FLOAT,
    x3 FLOAT,
    y3 FLOAT,
    x4 FLOAT,
    y4 FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);



-- Index for faster lookups by filename
CREATE INDEX IF NOT EXISTS idx_ocr_results_filename ON repo_ask.ocr_results(filename);

-- Comments on table and columns
COMMENT ON TABLE repo_ask.ocr_results IS 'Stores results from OCR processing';
COMMENT ON COLUMN repo_ask.ocr_results.filename IS 'Name of the processed image file';
COMMENT ON COLUMN repo_ask.ocr_results.ocr_text IS 'Full extracted text content';

-- table for storing Feedback requests
CREATE TABLE IF NOT EXISTS repo_ask.feedback_requests (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    ai_thinking TEXT,
    ai_answer TEXT,
    user_comments TEXT,
    feedback_type VARCHAR(20), -- 'accept' or 'reject'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Comments on table and columns
COMMENT ON TABLE repo_ask.feedback_requests IS 'Stores user feedback on AI responses';
COMMENT ON COLUMN repo_ask.feedback_requests.query IS 'The user query';
COMMENT ON COLUMN repo_ask.feedback_requests.ai_thinking IS 'The thinking process of the AI';
COMMENT ON COLUMN repo_ask.feedback_requests.ai_answer IS 'The answer provided by the AI';
COMMENT ON COLUMN repo_ask.feedback_requests.user_comments IS 'Optional comments from the user';
COMMENT ON COLUMN repo_ask.feedback_requests.feedback_type IS 'Type of feedback: accept or reject';