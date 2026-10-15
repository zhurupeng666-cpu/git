from src.rag.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

for event in pipeline.ask_stream(
    query="CVD腔体压力异常如何排查？",
    top_k=5
):
    print(event)

    #uvicorn src.api.main:app --reload