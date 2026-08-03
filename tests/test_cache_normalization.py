from app.utils.cache import _generate_cache_key

def test_cache_normalization():
    """
    Verifies that differently formatted versions of the same query
    map to the exact same cache key.
    """
    base_query = "What is the Transformer architecture?"
    
    # 1. Exact match
    key_base = _generate_cache_key(base_query)
    
    # 2. Extra whitespace and tabs
    key_whitespace = _generate_cache_key("   What    is \t the \n Transformer architecture?   ")
    
    # 3. Capitalization differences
    key_lower = _generate_cache_key("what is the transformer architecture?")
    key_upper = _generate_cache_key("WHAT IS THE TRANSFORMER ARCHITECTURE?")
    
    # Assert they all produce the identical cache key
    assert key_base == key_whitespace, "Whitespace normalization failed"
    assert key_base == key_lower, "Lowercase normalization failed"
    assert key_base == key_upper, "Uppercase normalization failed"
    
    # 4. Check version prefix
    assert key_base.startswith("llm_response:v1:"), "Cache key missing version prefix"

if __name__ == "__main__":
    test_cache_normalization()
    print("All cache normalization tests passed!")
