def get_retriever(vectorstore, k=5):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

def format_citations(docs):
    citations = []
    seen=set()
    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown")
        page=doc.metadata.get("page","?")
        key=f"{source}::{page}"
        if key not in seen:
            seen.add(key)
            citations.append({
                "file":source,
                "page": int(page)+1 if isinstance(page, int) else page,
                "snippet": doc.page_content[:200].strip() +"…",
            })
    return citations