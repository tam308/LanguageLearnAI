# LanguageLearnAI

This bot was built to facilitate easy language learning utilising google's genai python module. Configurable settings to tailor to any language required. Anki deck integration is also supported.

## Available commands

- `/start` - Start the bot
- `/profile` - Configure your profile information. Information given will be appended to the AI's system instructions.
- `/anki` - Gives you 5 random entries from your anki deck.

## Setup

1. Download or clone the repo, then install dependencies

   ```bash
   pip install -r requirements.txt
   ```

2. Configure API keys and additional info

   ```bash
   cp .env_example .env
   cp additional_language_rules_example.txt additional_language_rules.txt
   ```

   Then edit `.env` and fill in:
   - `TELEGRAM_TOKEN` from [@BotFather](https://t.me/BotFather)
   - `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey)
   - `OWNER` Your Telegram user ID. Use the [@userinfobot](https://t.me/userinfobot) to find your Telegram user ID.
   - `BOT_NAME`, `BOT_LANGUAGE`, `BOT_AGE` and `BOT_LOCATION` to give the bot its persona.

   The `additional_language_rules.txt` is optional and lets you give the bot extra personality or slang rules. Edit it to taste, or leave it as is.

## Exporting your Anki deck

Use https://jspann21.github.io/anki_excel_tool/, your csv file should mirror the example given and be named anki.csv

## Run

```bash
python languagelearn_AI.py
```

The bot restarts itself if it crashes, so it will keep running until you stop it. Once it is up, message it on Telegram to start chatting. A digest of 5 random Anki words is also sent automatically every day at 5:00 AM (GMT +8), and you can request one any time with `/anki`.

## Notes

AI may not be accurate, so take conversational grammar or corrections with a grain of salt.
The format of the Anki deck CSV may vary from deck to deck. If your columns don't line up, adjust the `WORD_COLUMN`, `DEFINITION_COLUMN`, `EXAMPLE_SENTENCE_COLUMN` and `TRANSLATION_COLUMN` values at the top of `languagelearn_AI.py`.
The daily reminder time and timezone are also configurable at the top of `languagelearn_AI.py`.