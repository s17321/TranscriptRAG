from app.pipelines.ingestion_pipeline import IngestionPipeline


def main() -> None:
    pipeline = IngestionPipeline()
    result = pipeline.run()

    print("Ingestion finished successfully.")
    print(f"Metadata rows: {result.metadata_rows}")
    print(f"Transcript files found: {result.transcript_files_found}")
    print(f"Documents created: {result.documents_created}")
    print(f"QA records loaded: {result.qa_records_loaded}")
    print(f"Missing transcripts: {len(result.missing_transcripts)}")
    print(f"QA without matching document: {len(result.qa_without_matching_document)}")


if __name__ == "__main__":
    main()