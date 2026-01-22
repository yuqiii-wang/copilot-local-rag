class RAGService:
    def get_dummy_search_results(self, query: str = ""):
        all_results = [
            {"title": "Confluence: Project Apollo Specs", "link": "https://confluence.example.com/pages/viewpage.action?pageId=12345"},
            {"title": "Jira: PROJ-123 - Backend API", "link": "https://jira.example.com/browse/PROJ-123"},
            {"title": "GitHub: Commit 8a2b3c - Fix login bug", "link": "https://github.com/example/repo/commit/8a2b3c4d"},
            {"title": "SharePoint: Q3 Roadmap.docx", "link": "https://company.sharepoint.com/sites/engineering/Shared%20Documents/Q3%20Roadmap.docx"},
        ]
        
        if not query:
            return all_results
            
        # Simple case-insensitive filter
        query_lower = query.lower()
        filtered = [r for r in all_results if query_lower in r['title'].lower()]
        
        # If no matches, return all (fallback) or return empty? 
        # Usually RAG returns most relevant. If nothing matches, returning nothing is correct.
        # But for valid demo, let's returns all if nothing matches, or maybe just the filtered list.
        # Let's return filtered list to prove search works.
        return filtered if filtered else all_results # Fallback to all if no match found for demo purposes

    def process_feedback(self, action: str, query: str, ai_thinking: str = "", ai_answer: str = "", comments: str = ""):
        # Log feedback to file or database
        print(f"RAG Feedback [{action}]: Query='{query[:50]}...', AI_Thinking='{len(ai_thinking)} chars', AI_Answer='{len(ai_answer)} chars', Comments='{comments}'")
        
        try:
            from pgdb.pgdb_manager import pg_manager
            sql = """
                INSERT INTO repo_ask.feedback_requests (query, ai_thinking, ai_answer, user_comments, feedback_type)
                VALUES (%s, %s, %s, %s, %s)
            """
            pg_manager.execute_query(sql, (query, ai_thinking, ai_answer, comments, action))
        except Exception as e:
            print(f"Error saving feedback: {e}")
            
        return {"action": action, "status": "processed"}

rag_service = RAGService()
