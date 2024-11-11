PWD=$(pwd)
docker cp $PWD/n8n_workflows/ n8n:/tmp/backups/latest/
docker cp $PWD/n8n_workflows/credentials.json n8n:/tmp/backups/latest/creds.json
docker exec n8n n8n import:credentials --input=/tmp/backups/latest/creds.json
docker exec n8n n8n import:workflow --separate --input=/tmp/backups/latest/
