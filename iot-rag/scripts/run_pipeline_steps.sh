# turn off chromadb sending stats back
export ANONYMIZED_TELEMETRY=False

BASEDIR=/Users/donohara/Data/50_HU/_github/hu_sp25_691_a03


# ----------------------------------
#  Step 01: Ingest: parse pdf or text files into cleaned text
# ----------------------------------
# python3 $BASEDIR/main.py step01_ingest --input_filename  Tanifuji_2021_Business_Domain.pdf
python3 $BASEDIR/main.py step01_ingest --input_filename all

# ----------------------------------
#  Step 02: Generate Embeddings from the cleaned text files
# ----------------------------------
# python3 $BASEDIR/main.py step02_generate_embeddings --input_filename Zhang_et_al_2024_LLMs_cleaned.txt
# python3 $BASEDIR/main.py step02_generate_embeddings --input_filename all

# ----------------------------------
#  Step 03: Store the cleaned text and embeddings in a vector db
# ----------------------------------
# python3 $BASEDIR/main.py step03_store_vectors  --input_filename Zhang_et_al_2024_LLMs_cleaned.txt
# python3 $BASEDIR/main.py step03_store_vectors --input_filename  all

# ----------------------------------
#  Step 04: Retrieve chunks of text and similarity scores
# ----------------------------------
# python3 $BASEDIR/main.py step04_retrieve_chunks --query_args "regional development in japan"

# ----------------------------------
#  Step 05: Run LLM Queries with and without RAG
#  	If the parameter "--use_rag" is not provided, RAG is not performed
# ----------------------------------
# QUERY="Discuss trends in regional development in Japan"
# python3 $BASEDIR/main.py step05_generate_response  --query_args "$QUERY"
# python3 $BASEDIR/main.py step05_generate_response  --query_args "$QUERY"  --use_rag
