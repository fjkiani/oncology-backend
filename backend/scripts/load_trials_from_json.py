# Using insert_one and catching duplicate key errors might be simpler if _id is guaranteed unique per run
# Or use upsert (depending on astrapy version/preference)
# Let's use simple insert and rely on _id uniqueness or script running once clean.
# If rerunning is needed, delete collection or use upsert logic.
# Update: Using upsert with the filter on _id is generally the robust approach for reruns.
result = collection.upsert(
    id=source_url,  # Use id parameter for document ID
    document=astra_document  # Pass full document
) 