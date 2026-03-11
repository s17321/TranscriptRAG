from app.pipelines.indexing_pipeline import IndexingPipeline


def main() -> None:
    pipeline = IndexingPipeline()
    result = pipeline.run()

    print("Indexing finished successfully.")
    print(f"Documents loaded: {result['documents_loaded']}")
    print(f"Chunks created: {result['chunks_created']}")
    print(f"Output file: {result['output_file']}")


if __name__ == "__main__":
    main()