import os
import re
import javalang
from pathlib import Path
from typing import List, Dict, Tuple
from services.knowledge_graph.tokenization_utils import identify_pattern

# Java language keywords to exclude (Still useful for regex fallback or general cleaning)
JAVA_KEYWORDS = {
    "abstract", "continue", "for", "new", "switch", "assert", "default", "goto", "package", "synchronized",
    "boolean", "do", "if", "private", "this", "break", "double", "implements", "protected", "throw",
    "byte", "else", "import", "public", "throws", "case", "enum", "instanceof", "return", "transient",
    "catch", "extends", "int", "short", "try", "char", "final", "interface", "static", "void",
    "class", "finally", "long", "strictfp", "volatile", "const", "float", "native", "super", "while",
    "true", "false", "null", "string", "system", "out", "println", "util", "java", "com"
}

def split_camel_snake(text: str) -> str:
    """
    Splits CamelCase and snake_case strings into space-separated words.
    """
    # Create space between camelCase: 'camelCase' -> 'camel Case'
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', text)
    # Handle consecutive caps: 'ABCCamel' -> 'ABC Camel'
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
    # Replace simple underscores and dashes
    return s2.replace('_', ' ').replace('-', ' ')

def clean_and_tokenize(text: str) -> List[str]:
    """
    Cleans text, removes Java keywords, handles Camel/Snake case, and tokenizes.
    This is used for NON-Java files or as a fallback.
    """
    # 1. Expand CamelCase (keep dashes/underscores distinct for numeric pattern analysis)
    text = re.sub(r'(.)([A-Z][a-z]+)', r'\1 \2', text)
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    
    raw_tokens = text.split()
    final_tokens_list = []
    
    for token in raw_tokens:
        # Check for specific patterns (UUID, ISIN, etc)
        pattern_tag = identify_pattern(token)
        matched = False
        
        if pattern_tag:
             # Add original token (cleaned)
             clean_t = re.sub(r'[^a-zA-Z0-9.,\-/]', '', token).lower()
             if clean_t:
                 final_tokens_list.append(clean_t)
                 # Append pattern name (e.g., 'price', 'isin', 'uuid') to enrich context
                 final_tokens_list.append(pattern_tag.replace('pattern_', ''))
                 matched = True
        
        if not matched:
            if any(c.isdigit() for c in token):            
                # Add cleaned number content
                cleaned_num = re.sub(r'[^a-zA-Z0-9.,\-/]', '', token).lower()
                if cleaned_num:
                    final_tokens_list.append(cleaned_num)
            else:
                # Text Token Logic
                t = token.replace('_', ' ').replace('-', ' ')
                # Allow dots to remain in tokens (e.g. com.dummy)
                t = re.sub(r'[^a-zA-Z0-9\s.]', '', t)
                final_tokens_list.extend(t.lower().split())
            
    # 4. Filter stop words (Java keywords + common English stops if needed)
    # Also filtering very short tokens, but keeping metadata tokens
    final_tokens = [
        t for t in final_tokens_list 
        if t not in JAVA_KEYWORDS and (len(t) > 2 or t.startswith('('))
    ]
    
    return final_tokens

def extract_java_tokens(source_code: str) -> List[str]:
    """
    Parses Java code using javalang to extract meaningful tokens:
    - Class/Interface names
    - Variable types and names (dependency)
    - Method names and invocations
    - String literals (logs)
    """
    tokens = []
    try:
        tree = javalang.parse.parse(source_code)
        
        for path, node in tree:
            # 1. Class/Interface Definitions
            if isinstance(node, (javalang.tree.ClassDeclaration, javalang.tree.InterfaceDeclaration)):
                tokens.append(node.name)
                # Extends/Implements used as types are captured in specific node fields or generally in Type nodes
                if hasattr(node, 'extends') and node.extends:
                    if isinstance(node.extends, list):
                        for ext in node.extends:
                             if hasattr(ext, 'name'): tokens.append(ext.name)
                    elif hasattr(node.extends, 'name'):
                        tokens.append(node.extends.name)
                if hasattr(node, 'implements') and node.implements:
                    for imp in node.implements:
                        if hasattr(imp, 'name'): tokens.append(imp.name)

            # 2. Fields (Variables defined in class)
            elif isinstance(node, javalang.tree.FieldDeclaration):
                if node.type and hasattr(node.type, 'name'):
                    tokens.append(node.type.name) # Dependency on Type
                for declarator in node.declarators:
                    tokens.append(declarator.name) # Variable Name

            # 3. Local Variables
            elif isinstance(node, javalang.tree.LocalVariableDeclaration):
                 if node.type and hasattr(node.type, 'name'):
                     tokens.append(node.type.name)
                 for declarator in node.declarators:
                     tokens.append(declarator.name)

            # 4. Method Definitions
            elif isinstance(node, javalang.tree.MethodDeclaration):
                tokens.append(node.name)
                # Return type
                if node.return_type and hasattr(node.return_type, 'name'):
                    tokens.append(node.return_type.name)

            # 5. Method Invocations
            elif isinstance(node, javalang.tree.MethodInvocation):
                 tokens.append(node.member)

            # 6. Class Instantiation (new ClassName)
            elif isinstance(node, javalang.tree.ClassCreator):
                 if node.type and hasattr(node.type, 'name'):
                     tokens.append(node.type.name)

            # 7. String Literals (Potential Logs)
            elif isinstance(node, javalang.tree.Literal):
                # javalang returns value as string including quotes: '"Error"'
                val = str(node.value)
                if val.startswith('"') and val.endswith('"'):
                    content = val[1:-1]
                    # Tokenize the content of the string
                    sub_tokens = clean_and_tokenize(content)
                    tokens.extend(sub_tokens)
    
    except Exception as e:
        # Fallback to regex if parsing fails
        # print(f"Warning: Javalang parse failed, falling back to regex. Error: {e}")
        return clean_and_tokenize(source_code)
        
    # Post-process: Clean extracted tokens using basic rules (camelCase splitting etc)
    # Because javalang gives "CamelCase", we might want "Camel", "Case" too?
    # The requirement says "extract every word". 
    # If we return 'TradingSystem', the vectorizer might just count 'TradingSystem'.
    # If we want 'Trading' and 'System', we need to split.
    
    expanded_tokens = []
    for t in tokens:
        # Split camel/snake case
        cleaned_str = split_camel_snake(t)
        # Split by space and add
        sub = cleaned_str.split()
        for s in sub:
            if s.lower() not in JAVA_KEYWORDS and len(s) > 2:
                expanded_tokens.append(s.lower())
                
    return expanded_tokens

def load_documents(dataset_path: str) -> List[Dict]:
    """
    Walks through the dataset directory and processes files (Confluence, Java, Jira).
    Returns a list of document objects (id, type, content, tokens).
    """
    documents = []
    base_path = Path(dataset_path)
    
    if not base_path.exists():
        print(f"Dataset path {dataset_path} does not exist.")
        return []

    # Iterate over all files
    for file_path in base_path.rglob('*'):
        if file_path.is_file():
            # Determine type based on parent folder or extension
            doc_type = "unknown"
            if "confluence" in str(file_path).lower() or file_path.suffix == '.txt':
                doc_type = "confluence"
            if "jira" in str(file_path).lower():
                doc_type = "jira"
            if file_path.suffix == '.java':
                doc_type = "code"
            
            # Read content
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Use specialized tokenizer for Java
                if doc_type == "code":
                    tokens = extract_java_tokens(content)
                else:
                    tokens = clean_and_tokenize(content)
                
                # Only add if we have content
                if tokens:
                    documents.append({
                        "id": str(file_path.relative_to(base_path)),
                        "path": str(file_path),
                        "type": doc_type,
                        "raw_content": content,
                        "tokens": tokens
                    })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return documents

if __name__ == "__main__":
    # Test execution
    current_dir = Path(__file__).parent
    dataset_dir = current_dir / "dummy_dataset"
    docs = load_documents(str(dataset_dir))
    print(f"Loaded {len(docs)} documents.")
    if docs:
        all_tokens = [token for doc in docs for token in doc['tokens']]
        unique_tokens = set(all_tokens)
        print(f"Total number of tokens: {len(all_tokens)}")
        print(f"Total number of unique tokens: {len(unique_tokens)}")
        print(f"Example tokens from {docs[0]['id']}: {docs[0]['tokens'][:10]}")
