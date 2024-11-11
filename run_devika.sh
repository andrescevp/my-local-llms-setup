BASE_DIR=$(pwd)/devika-official-repo

if [ ! -d $BASE_DIR ]; then
  git clone https://github.com/stitionai/devika ./devika-official-repo
fi

cd $BASE_DIR
# set python version
pyenv local 3.12

# check venv exists
if [ ! -d "venv" ]; then
  python -m venv venv
fi

. venv/bin/activate
chmod +x setup.sh
./setup.sh
cd $BASE_DIR
# start devika
python devika.py | cd ui && bun run start