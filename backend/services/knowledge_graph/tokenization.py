import os
import re
import javalang
from pathlib import Path
from typing import List, Dict, Tuple
from html.parser import HTMLParser
from services.knowledge_graph.tokenization_utils import identify_pattern, extract_capital_sequences, generate_ngrams
from services.knowledge_graph import feature_weights

# Java language keywords to exclude (Still useful for regex fallback or general cleaning)
JAVA_KEYWORDS = {
    "abstract", "continue", "for", "new", "switch", "assert", "default", "goto", "package", "synchronized",
    "boolean", "do", "if", "private", "this", "break", "double", "implements", "protected", "throw",
    "byte", "else", "import", "public", "throws", "case", "enum", "instanceof", "return", "transient",
    "catch", "extends", "int", "short", "try", "char", "final", "interface", "static", "void",
    "class", "finally", "long", "strictfp", "volatile", "const", "float", "native", "super", "while",
    "true", "false", "null", "string", "system", "out", "println", "util", "java", "com"
}

# C++ Keywords
CPP_KEYWORDS = {
    "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit", "atomic_noexcept", "auto",
    "bitand", "bitor", "bool", "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t", "class",
    "compl", "concept", "const", "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
    "co_return", "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast", "else", "enum",
    "explicit", "export", "extern", "false", "float", "for", "friend", "goto", "if", "inline", "int", "long",
    "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private",
    "protected", "public", "register", "reinterpret_cast", "requires", "return", "short", "signed", "sizeof",
    "static", "static_assert", "static_cast", "struct", "switch", "template", "this", "thread_local", "throw",
    "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual", "void", "volatile",
    "wchar_t", "while", "xor", "xor_eq", "std", "include", "define", "ifdef", "endif"
}

# Bash Keywords
BASH_KEYWORDS = {
    "if", "then", "else", "elif", "fi", "case", "esac", "for", "select", "while", "until", "do", "done", "in",
    "function", "time", "coproc", "declare", "typeset", "local", "readonly", "export", "unset", "set", "shift",
    "trap", "ulimit", "umask", "alias", "unalias", "break", "continue", "return", "exit", "echo", "printf",
    "read", "cd", "pwd", "pushd", "popd", "dirs", "eval", "exec", "source", "true", "false", "null"
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
    # Also filtering very short tokens, but keeping metadata tokens and numbers
    final_tokens = [
        t for t in final_tokens_list 
        if t not in JAVA_KEYWORDS and (len(t) > 2 or t.startswith('(') or any(c.isdigit() for c in t))
    ]
    
    return final_tokens

def tokenize_log_message(text: str) -> List[str]:
    """
    Standard tokenization for Log/Error Strings.
    """
    tokens = clean_and_tokenize(text)
    return tokens

class StructuralHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.segments = [] # List of (tokens, tags)
        self.current_tags = []

    def handle_starttag(self, tag, attrs):
        attr_dict = {k: v for k, v in attrs} if attrs else {}
        self.current_tags.append({'name': tag, 'attrs': attr_dict})

    def handle_endtag(self, tag):
        # find the last occurrence of tag
        for i in range(len(self.current_tags) - 1, -1, -1):
            if self.current_tags[i]['name'] == tag:
                del self.current_tags[i]
                break

    def handle_data(self, data):
        # Ignore content within script and style tags to avoid CSS/JS noise
        if any(tag['name'] in ('script', 'style', 'noscript', 'iframe', 'svg') for tag in self.current_tags):
            return

        tokens = clean_and_tokenize(data)
        if tokens:
            self.segments.append((tokens, [t.copy() for t in self.current_tags]))

def extract_html_segments(text: str) -> List[Tuple[List[str], List[Dict]]]:
    """
    Parses HTML and returns a list of segments, where each segment is (tokens, tags).
    """
    parser = StructuralHTMLParser()
    try:
        parser.feed(text)
        return parser.segments
    except Exception:
        # Fallback
        return [(clean_and_tokenize(text), [])]

def extract_html_tokens(text: str) -> List[str]:
    """
    Extracts tokens from HTML content with structural weighting.
    
    This function:
    1. Parses HTML into segments (text content + context tags).
    2. Calculates a weight for each segment based on tags (h1, strong, etc.) using feature_weights.
    3. Repeats tokens based on weight to emphasize important sections.
    4. Generates n-grams for high-weight sections to capture phrase context.
    """
    segments = extract_html_segments(text)
    
    # Dictionary to hold token layers. layer[i] will hold tokens for the i-th repetition.
    # This prevents "Word Word" artifacts by preserving the sequence in each layer.
    layers = {} 
    explicit_ngrams = []
    max_layer = 0

    for tokens, tags in segments:
        if not tokens:
            continue
            
        multiplier = 1.0
        
        # Calculate maximum weight from the context stack
        for tag_info in tags:
            # Check for specific tag weights functionality
            # tag_info contains {'name': 'tag', 'attrs': {...}}
            w = feature_weights.calculate_html_weight(tag_info, len(tokens))
            if w > multiplier:
                multiplier = w
                
        # Round to integer for repetition
        count = int(round(multiplier))
        
        if count > max_layer:
            max_layer = count

        # Populate layers
        for c in range(1, count + 1):
            if c not in layers:
                layers[c] = []
            layers[c].extend(tokens)
            
        # 2. Add structural N-grams for important sections
        # If a headers/title/bold section, the phrase itself is a key feature (e.g., "Risk Policy")
        # We add bigrams and trigrams.
        if count >= 3 and len(tokens) > 1:
            ngrams = generate_ngrams(tokens, n_min=2, n_max=3)
            # Add ngrams with half the weight of individual tokens (heuristic) or full weight?
            # We treat them as highly specific identifiers
            for _ in range(count):
                explicit_ngrams.extend(ngrams)
                
    # Flatten layers into a single sequence
    weighted_tokens = []
    
    for c in range(1, max_layer + 1):
        if c in layers:
            weighted_tokens.extend(layers[c])
            
    weighted_tokens.extend(explicit_ngrams)
    
    return weighted_tokens

def extract_java_tokens(source_code: str) -> List[str]:
    """
    Parses Java code using javalang to extract meaningful tokens:
    - Class/Interface names
    - Variable types and names (dependency)
    - Method names and invocations
    - String literals (logs)
    """
    identifiers = []
    literal_tokens = []
    try:
        tree = javalang.parse.parse(source_code)
        
        for path, node in tree:
            # 1. Class/Interface Definitions
            if isinstance(node, (javalang.tree.ClassDeclaration, javalang.tree.InterfaceDeclaration)):
                identifiers.append(node.name)
                # Extends/Implements used as types are captured in specific node fields or generally in Type nodes
                if hasattr(node, 'extends') and node.extends:
                    if isinstance(node.extends, list):
                        for ext in node.extends:
                             if hasattr(ext, 'name'): identifiers.append(ext.name)
                    elif hasattr(node.extends, 'name'):
                        identifiers.append(node.extends.name)
                if hasattr(node, 'implements') and node.implements:
                    for imp in node.implements:
                        if hasattr(imp, 'name'): identifiers.append(imp.name)

            # 2. Fields (Variables defined in class)
            elif isinstance(node, javalang.tree.FieldDeclaration):
                if node.type and hasattr(node.type, 'name'):
                    identifiers.append(node.type.name) # Dependency on Type
                for declarator in node.declarators:
                    identifiers.append(declarator.name) # Variable Name

            # 3. Local Variables
            elif isinstance(node, javalang.tree.LocalVariableDeclaration):
                 if node.type and hasattr(node.type, 'name'):
                     identifiers.append(node.type.name)
                 for declarator in node.declarators:
                     identifiers.append(declarator.name)

            # 4. Method Definitions
            elif isinstance(node, javalang.tree.MethodDeclaration):
                identifiers.append(node.name)
                # Return type
                if node.return_type and hasattr(node.return_type, 'name'):
                    identifiers.append(node.return_type.name)

            # 5. Method Invocations
            elif isinstance(node, javalang.tree.MethodInvocation):
                 identifiers.append(node.member)

            # 6. Class Instantiation (new ClassName)
            elif isinstance(node, javalang.tree.ClassCreator):
                 if node.type and hasattr(node.type, 'name'):
                     identifiers.append(node.type.name)

            # 7. String Literals (Potential Logs)
            elif isinstance(node, javalang.tree.Literal):
                # javalang returns value as string including quotes: '"Error"'
                val = str(node.value)
                if val.startswith('"') and val.endswith('"'):
                    content = val[1:-1]
                    # Tokenize the content of the string - these are already clean and should NOT be split further if unique
                    sub_tokens = clean_and_tokenize(content)
                    literal_tokens.extend(sub_tokens)
    
    except Exception as e:
        # Fallback to regex if parsing fails
        # print(f"Warning: Javalang parse failed, falling back to regex. Error: {e}")
        return clean_and_tokenize(source_code)
        
    # Post-process: Clean extracted tokens using basic rules
    
    expanded_tokens = []
    
    # Process Identifiers (Split CamelCase/SnakeCase)
    for t in identifiers:
        # Split camel/snake case
        cleaned_str = split_camel_snake(t)
        # Split by space and add
        sub = cleaned_str.split()
        for s in sub:
            if s.lower() not in JAVA_KEYWORDS and (len(s) > 2 or any(c.isdigit() for c in s)):
                expanded_tokens.append(s.lower())

    # Add Literal Tokens directly (they obey clean_and_tokenize rules, preserving dash-ids like AAPL-OCT-160-C)
    expanded_tokens.extend(literal_tokens)
                
    return expanded_tokens

def extract_cpp_tokens(source_code: str) -> List[str]:
    """
    Parses C++ code using regex to extract meaningful tokens:
    - #include headers
    - Class/Struct names
    - Function names
    - Variable types
    - String literals
    """
    identifiers = []
    literal_tokens = []
    
    # 1. Includes
    includes = re.findall(r'#include\s*[<"]([^>"]+)[>"]', source_code)
    identifiers.extend(includes)
    
    # 2. Class/Struct Definitions (class MyClass, struct MyStruct)
    class_defs = re.findall(r'\b(class|struct)\s+(\w+)', source_code)
    for _, name in class_defs:
        identifiers.append(name)
        
    # 3. Function Definitions/Declarations (Type FuncName(...))
    # Heuristic: Word followed by (
    func_calls = re.findall(r'\b(\w+)\s*\(', source_code)
    identifiers.extend(func_calls)
    
    # 4. Types/Variables (Type var;) - Hard to do perfectly with regex, picking capitalized words as potential Types
    # and words before equals
    
    # 5. String Literals
    strings = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', source_code)
    for s in strings:
        # Tokenize content of string literals (preserve dash-ids)
        literal_tokens.extend(clean_and_tokenize(s))
        
    # General cleanup and tokenization of the rest of the code to catch variables etc
    # Remove comments
    no_comments = re.sub(r'//.*', '', source_code)
    no_comments = re.sub(r'/\*.*?\*/', '', no_comments, flags=re.DOTALL)
    
    # Extract words (identifiers)
    words = re.findall(r'\b[a-zA-Z_]\w*\b', no_comments)
    identifiers.extend(words)
    
    expanded_tokens = []
    
    # Process Identifiers
    for t in identifiers:
        cleaned_str = split_camel_snake(t)
        sub = cleaned_str.split()
        for s in sub:
            s_lower = s.lower()
            if s_lower not in CPP_KEYWORDS and (len(s_lower) > 2 or any(c.isdigit() for c in s_lower)):
                expanded_tokens.append(s_lower)

    # Add Literals direct
    expanded_tokens.extend(literal_tokens)

    return expanded_tokens

def extract_bash_tokens(source_code: str) -> List[str]:
    """
    Parses Bash scripts using regex to extract meaningful tokens:
    - Variable assignments
    - Function definitions
    - Commands
    """
    identifiers = []
    literal_tokens = []
    
    # 1. Variable Assignments (VAR=val)
    vars = re.findall(r'\b([a-zA-Z_]\w*)=', source_code)
    identifiers.extend(vars)
    
    # 2. Function Definitions (func() or function func)
    funcs = re.findall(r'\bfunction\s+(\w+)', source_code)
    funcs2 = re.findall(r'\b(\w+)\s*\(\)', source_code)
    identifiers.extend(funcs)
    identifiers.extend(funcs2)
    
    # 3. String Literals
    strings = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', source_code)
    strings_single = re.findall(r"'([^']*)'", source_code)
    
    for s in strings + strings_single:
         literal_tokens.extend(clean_and_tokenize(s))
    
    # General extraction of words, filtering typical shell syntax
    # Remove comments
    no_comments = re.sub(r'#.*', '', source_code)
    
    words = re.findall(r'\b[a-zA-Z0-9_.-]+\b', no_comments)
    identifiers.extend(words)
    
    expanded_tokens = []
    for t in identifiers:
        # Bash variables often use snake_case or CAPS
        cleaned_str = split_camel_snake(t)
        sub = cleaned_str.split()
        for s in sub:
            s_lower = s.lower()
            # Remove extension-like tokens if they are just extensions? No, keep everything unless keyword
            if s_lower not in BASH_KEYWORDS and (len(s_lower) > 2 or any(c.isdigit() for c in s_lower)):
                # Remove leading/trailing formatting chars
                s_clean = s_lower.strip('.-_')
                if len(s_clean) > 1 or any(c.isdigit() for c in s_clean):
                    expanded_tokens.append(s_clean)

    # Add Literals
    expanded_tokens.extend(literal_tokens)
                    
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
            tokenizer = clean_and_tokenize # Default

            if "confluence" in str(file_path).lower() or file_path.suffix == '.txt':
                doc_type = "confluence"
            if "jira" in str(file_path).lower():
                doc_type = "jira"
            
            # Code detection - Specific tokenizers override default
            if file_path.suffix == '.java':
                doc_type = "code"
                tokenizer = extract_java_tokens
            elif file_path.suffix in ['.cpp', '.cc', '.cxx', '.c', '.h', '.hpp']:
                doc_type = "code"
                tokenizer = extract_cpp_tokens
            elif file_path.suffix in ['.sh', '.bash']:
                doc_type = "code"
                tokenizer = extract_bash_tokens
            elif file_path.suffix == '.sql':
                doc_type = "code"
                # Use default clean_and_tokenize for now, or add extract_sql_tokens later
                
            # Read content
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                tokens = tokenizer(content)
                
                # Check for All-Caps Phrases (Entities) and stick them into the token list
                capital_phrases = extract_capital_sequences(content)
                if capital_phrases:
                    # Treat them as single tokens (underscored) or just add them?
                    # Adding them as n-grams will happen in FG if we concat, 
                    # but if we add them as separate tokens, they get indexed as unigrams "RISK MANAGEMENT POLICY"
                    # which is great for "Exact Match" features.
                    tokens.extend(capital_phrases)

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
