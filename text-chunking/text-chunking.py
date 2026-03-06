def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    stride = chunk_size - overlap
    chunks = []
    
    for i in range(0, len(tokens), stride):
        chunk = tokens[i : i + chunk_size]
        chunks.append(chunk)
        

        if i + chunk_size >= len(tokens):
            break
            
    return chunks