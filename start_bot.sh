#!/bin/bash
#rebuilds the image and (re)starts the bot. run this after editing code OR .env
#secrets (.env) and your data files stay on the server and are passed in at runtime, not built into the image
cd "$(dirname "$0")" || exit 1

#docker's --env-file keeps quotes and windows line endings literally (python-dotenv strips them), so clean them here
ENV_VARS=$(sed -E "s/\r$//; s/^([A-Za-z_][A-Za-z0-9_]*)=([\"'])(.*)\2[[:space:]]*$/\1=\3/" .env) || exit 1

#bind-mounted files must exist first, or docker creates empty folders with these names
touch history.json anki.csv user_profile.txt additional_language_rules.txt

echo "Building new Docker image with latest code..."
docker build -t language-bot . || exit 1

echo "Stopping the old bot..."
docker stop running-lang-bot 2>/dev/null
docker rm running-lang-bot 2>/dev/null

echo "Starting the updated bot..."
docker run -d \
  --name running-lang-bot \
  --restart unless-stopped \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --log-opt max-size=10m --log-opt max-file=3 \
  --env-file <(printf '%s\n' "$ENV_VARS") \
  -v "$(pwd)/history.json:/app/history.json" \
  -v "$(pwd)/anki.csv:/app/anki.csv" \
  -v "$(pwd)/user_profile.txt:/app/user_profile.txt" \
  -v "$(pwd)/additional_language_rules.txt:/app/additional_language_rules.txt" \
  language-bot || exit 1

echo "Bot updated and running in the background!"
