PWD=$(pwd)
docker exec n8n n8n export:workflow --backup --output=/tmp/backups/latest/
docker cp n8n:/tmp/backups/latest/ $PWD/n8n_workflows/
docker exec n8n n8n export:credentials --all --output=/tmp/backups/latest/creds.json
docker cp n8n:/tmp/backups/latest/creds.json $PWD/n8n_workflows/credentials.json
