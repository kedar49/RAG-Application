### 1. Run PgVector 
### these commands i've also added in `run_pgvector.sh` bash file

- Run using this script

```shell
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
   phidata/pgvector:16
```

### 2. Run RAG app

```shell
streamlit run app.py
```
