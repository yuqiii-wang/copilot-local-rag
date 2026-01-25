import sys
import os

# Add current dir to path
sys.path.append(os.getcwd())

from services.knowledge_graph.java_code_indexer import JavaCodeIndexer

def test_indexer():
    print("Running Indexer Test...")
    root = "services/knowledge_graph/dummy_dataset/dummy_trading_sys_java/src" # Correct relative path
    # But wait, test files are not in src?
    # The structure is:
    # dummy_trading_sys_java/src/...
    # dummy_trading_sys_java/test/...
    
    # If I only point to 'src', I won't scan OptionTest.java (which is in test)
    # The previous code in train_model.py pointed to:
    # code_root = os.path.join("services", "knowledge_graph", "dummy_dataset", "dummy_trading_sys_java", "src")
    
    # BUT OptionTest is in 'test'. So the indexer WAS NOT scanning OptionTest.java ??
    # If the indexer was only scanning src, then it never parsed OptionTest, so it never found "USOPTIONS4" in the literal analysis.
    
    # Let's target the parent folder of both src and test
    root = "services/knowledge_graph/dummy_dataset/dummy_trading_sys_java"
    
    indexer = JavaCodeIndexer(root)
    indexer.index_project()
    priors = indexer.get_code_priors()
    
    print(f"Generated {len(priors)} code priors.")

    print(f"Class -> File Map: {len(indexer.class_to_file)} classes found.")
    if "DummyTestBase" in indexer.class_to_file:
        print(f"DummyTestBase found in: {indexer.class_to_file['DummyTestBase']}")
    else:
        print("DummyTestBase NOT found in class_to_file")
        
    if "OptionTest.java" in indexer.pending_deps:
        print(f"OptionTest dependencies: {indexer.pending_deps['OptionTest.java']}")
    else:
        print("OptionTest has no pending deps")

    token = "usoptions4"
    if token in priors:
        print(f"Token '{token}' priors:")
        for filename, weight in priors[token].items():
            print(f"  - {filename}: {weight}")
    else:
        print(f"Token '{token}' NOT found in priors")
        print("Available tokens sample:", list(priors.keys())[:10])

if __name__ == "__main__":
    test_indexer()
