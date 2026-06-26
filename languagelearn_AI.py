import asyncio
import datetime
import io
import json
import logging
import os
import random
import time
import pandas as pd
import pytz
from dotenv import load_dotenv
from google import genai
from google.genai import types
from telegram import Update
from telegram.error import BadRequest, NetworkError
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

#resources
#https://ai.google.dev/gemini-api/docs

#config for anki, these are the column positions in the Core 2000 export and they start from zero
WORD_COLUMN = 2            #the vocab word with furigana, like 一[ひと]つ where the reading sits in brackets
DEFINITION_COLUMN = 4      #the english meaning
EXAMPLE_SENTENCE_COLUMN = 9   #example sentence with furigana, use 8 if you want the plain version without furigana
TRANSLATION_COLUMN = 11    #the english translation of the example sentence

#timezone for the daily anki reminder, you can use any IANA name like Asia/Singapore or Europe/London
TIMEZONE = "Asia/Singapore"
#hour of the day to send the daily reminder, in 24 hour time
REMINDER_HOUR = 5
#max tokens the model can use per reply, this is a hard limit, so if the model is verbose it may cut off the reply before it finishes
MAX_OUTPUT_TOKENS = 2000
#chance from 0 to 1 that a normal reply is nudged to be short and quick like a real text, this keeps casual chat feeling human
SHORT_REPLY_CHANCE = 0.4
#how many times to try sending a telegram message before giving up, and how many seconds to wait between tries
MAX_TELEGRAM_SEND_RETRY = 10
MAX_TELEGRAM_RETRY_DELAY = 10

#pull the API keys from the .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OWNER = os.getenv("OWNER") #telegram user ID of the bot owner, used to restrict access to the bot to only the owner
BOT_NAME = os.getenv("BOT_NAME")
BOT_LANGUAGE = os.getenv("BOT_LANGUAGE")
BOT_AGE = os.getenv("BOT_AGE")
BOT_LOCATION = os.getenv("BOT_LOCATION")

#history management functions to load and save chat history to a JSON file
#use utf-8 text encoding so Japanese and other characters save properly
def load_history(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        #only accept a list, anything else like corrupted JSON is treated as empty history
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []

def save_history(history, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
    except OSError as e:
        logging.error(f"Failed to save history to {file_path}: {e}")

#load the user's plain text profile, or an empty string if none has been saved yet
def load_profile():
    try:
        with open("user_profile.txt", "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""

#load the additional language and personality rules, or an empty string if the file is missing
def load_language_rules():
    try:
        with open("additional_language_rules.txt", "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""

#display logging messages in the console for debugging purposes
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.WARNING
)

#start command handler that sends a welcome message to the user when they start the bot
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #send a welcome message to the user when they start the bot
    await update.message.reply_text(f"Hello! I'm {BOT_NAME}, your friendly {BOT_LANGUAGE} learning companion. I'm here to help you practice and improve your {BOT_LANGUAGE} skills. Simply send me any messages in {BOT_LANGUAGE}, and I'll respond in kind.\n\n"
                                    f"First, please tell me a little about yourself, using the /profile command. This will help me tailor our conversations to your needs.\n\n "
                                    f"Minimally, please tell me your name, your current {BOT_LANGUAGE} level (beginner, intermediate, advanced), and what you hope to achieve by learning {BOT_LANGUAGE}. Anything else you " 
                                    f"share about your interests, hobbies, or preferred topics of conversation will "
                                    f"help me make our chats more engaging and relevant to you. This will be saved to the bots memory, and can be overwritten at any time by using the /profile command again.")

#profile command handler that saves the user's profile information to a file
async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #get the user's profile information from the command arguments
    message = " ".join(context.args)
    if not message:
        await update.message.reply_text("Please provide your profile information after the /profile command. For example:\n\n"
                                        f"/profile My name is Jason, I'm an expert {BOT_LANGUAGE} learner, and I want to improve my conversational skills. I like detective novels and playing games.")
        return
    #save the user's raw profile text to a file (the framing label is added when it is loaded into the system instruction)
    user_profile = message
    try:
        with open("user_profile.txt", "w", encoding="utf-8") as f:
            f.write(user_profile)
        logging.warning("User profile saved to user_profile.txt")
    except OSError as e:
        logging.error(f"Failed to save profile: {e}")
        await update.message.reply_text("Sorry, I couldn't save your profile right now. Please try again.")
        return

    await update.message.reply_text("Thanks! Your profile information has been saved. I will use this information to tailor our conversations to your needs.")

#send a telegram message, if the proxy or network drops we wait and try again, this happens often on hosts like PythonAnywhere
async def send_with_retry(target_message, text, attempts=MAX_TELEGRAM_SEND_RETRY, delay=MAX_TELEGRAM_RETRY_DELAY):
    for attempt in range(attempts):
        try:
            await target_message.reply_text(text)
            return
        except BadRequest:
            raise  #a malformed request will not fix itself so do not retry it
        except NetworkError as e:
            if attempt < attempts - 1:
                logging.warning(f"Telegram send failed, retrying in {delay} seconds: {e}")
                await asyncio.sleep(delay)
            else:
                logging.error(f"Gave up sending a telegram message after {attempts} attempts: {e}")

#The message command handler that processes incoming messages, sends them to the Gemini API for response generation, and replies back to the user
async def message_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message.text

    #build the client and set up the model and tools for generating content
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-flash-lite-latest"
    #build the system instruction, and append the saved profile so the model can personalize its responses
    system_instruction = [
            types.Part.from_text(text=f"""You are a friendly, supportive male {BOT_LANGUAGE} friend to the user. Your primary goal is to maintain a natural, engaging conversation while gently aiding the user's {BOT_LANGUAGE} learning journey.

Core Persona:
- Your name is {BOT_NAME}.
- You speak casually but grammatically correctly (e.g., standard plain form / dictionary form, casual polite forms when appropriate for a friend).
- You are playful, warm, and have a good sense of humor. You text like a real friend chatting on LINE or any messaging app: expressive, relaxed, and fun, while still sounding like a normal, well-educated {BOT_AGE}-year-old university student in {BOT_LOCATION}.
- You tease gently and joke around, but you are always encouraging and never mean-spirited.
- Your tone is warm, encouraging, and patient.
- You respond primarily in {BOT_LANGUAGE}. You only use English when the user explicitly asks for an explanation of a complex grammar concept or cultural nuance that would be too difficult to explain in {BOT_LANGUAGE}.

LearnLM Educational Guidelines (STRICT):
1. Accuracy is Paramount: Never fabricate grammar rules, vocabulary definitions, or cultural facts. If you are unsure, state that you are unsure.
2. Scaffolding, Not Solving: When the user struggles to express something, provide the missing vocabulary word or a gentle hint, rather than rewriting their entire thought for them immediately. 
3. Error Correction: 
   - DO NOT correct every minor particle mistake or slight unnatural phrasing. Prioritize conversational flow.
   - ONLY correct errors if they completely obscure the meaning, or if the user uses a fundamentally incorrect grammar structure (e.g., confusing transitive/intransitive verbs in a way that breaks the sentence).
   - When correcting, do so smoothly and encouragingly. (e.g., \"Ah, you mean [Corrected {BOT_LANGUAGE}]? Yeah, I agree!\"). Do not sound like a textbook or a strict teacher. 

Formatting:
- For normal back and forth conversation, reply like a real person texting: usually just one short message, and two at the very most. Do not split casual chitchat into many messages, it feels spammy and unnatural.
- Only when you are giving a fuller explanation or a correction followed by a separate question should you use multiple messages. In that case, put each distinct part on its own line with a line break between them, and each part is sent to the user as its own message.
- If you must explain a concept in English, clearly separate it from the conversational {BOT_LANGUAGE}."""),
    ]
    #add the additional language and personality rules if present
    language_rules = load_language_rules()
    if language_rules:
        system_instruction.append(types.Part.from_text(text=language_rules))
    #read the user's saved profile and add it to the system instruction (with a label so the model knows what it is)
    user_profile = load_profile()
    if user_profile:
        system_instruction.append(types.Part.from_text(text="The following information was provided about the user:\n\n" + user_profile))

    #sometimes nudge the model to keep a normal reply short and quick so the bot feels like a real person texting
    if random.random() < SHORT_REPLY_CHANCE:
        system_instruction.append(types.Part.from_text(text="For this reply only, if this is just casual conversation, answer in a very short and quick way like a real person texting, sometimes only a few words, and do not ask a follow up question. If the user is asking for help, a correction, or an explanation, ignore this and reply normally and fully."))

    generate_content_config = types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(
            thinking_level="MEDIUM",
        ),
        system_instruction=system_instruction,
    )
    #load saved chat history and restore it into SDK Content objects
    records = load_history("history.json")
    history = [
        types.Content(role=r["role"], parts=[types.Part.from_text(text=r["text"])])
        for r in records
        if isinstance(r, dict) and "role" in r and "text" in r  #skip any malformed records
    ]

    #create a new chat session with the model and send the user's message to it, then reply with the model's response
    try:
        logging.warning("Message received by model... Generating...") #warning so it shows up
        chat = client.chats.create(
            model=model,
            config=generate_content_config,
            history=history
        )
        #send the user's message to the model and get the response
        response = await asyncio.to_thread(chat.send_message, message)
        reply = response.text

        #append this turn to the history and save it before we send anything back
        if reply:
            records.append({"role": "user", "text": message})
            records.append({"role": "model", "text": reply})
            if len(records) > 80:  #keep only the last 80 messages for context
                records = records[-80:]
            save_history(records, "history.json")

            print(f"User: {message}")
            print(f"{BOT_NAME}: {reply}")

        #send the reply back, the model may split it with line breaks so send each line as its own message
        if reply:
            for part in reply.split("\n"): #split into parts so its more natural like texting
                part = part.strip()
                if part: #dont accidentally send empty lines
                    await send_with_retry(update.message, part)
        else:
            logging.warning("Model returned an empty response.")
            await send_with_retry(update.message, "Sorry, I couldn't generate a response at this time.")

    except Exception as e:
        logging.error(f"Error in message_command: {e}")
        try:
            await update.message.reply_text("Sorry, I encountered an error while processing your request.")
        except Exception:
            pass  #ignore failures to send the error message, like if the user has blocked the bot
        return

#manual command trigger for the Anki integration, which sends 5 random words from anki.csv to the user
async def anki_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.job_queue is None:
        await update.message.reply_text("Job queue is not available, so the Anki feature is disabled.")
        return
    context.job_queue.run_once(daily_reminder, when=0, data=update.effective_chat.id, name="anki_test")

async def daily_reminder(context: ContextTypes.DEFAULT_TYPE) -> None:
    #this runs the anki digest, the grace time lets it still fire if the bot was busy at the exact moment
    user_id = context.job.data
    context.job_queue.run_once(anki_integration, when=0, chat_id=user_id, data=user_id,
                                job_kwargs={"misfire_grace_time": 60})
        
#give the user 5 random words from anki.csv, it prints the word, definition, example sentence and translation
#these are sent as 5 separate messages so the user can easily copy one and ask the AI to explain it
async def anki_integration(context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await context.bot.send_message(chat_id=context.job.chat_id, text="Here are 5 random words from your Anki deck for you to study today:")
        #the Anki export begins with metadata lines starting with '#' (and has no header row),
        #so skip every '#' line (allowing leading " because some UIDs have it) so the data truly begins at the entries
        with open("anki.csv", encoding="utf-8-sig") as f:
            data_lines = [ln for ln in f if not ln.lstrip().lstrip('"').startswith("#")]
        if not data_lines:
            await context.bot.send_message(chat_id=context.job.chat_id, text="The Anki CSV file is empty or not found.")
            return
        df = pd.read_csv(io.StringIO("".join(data_lines)), header=None)
        random_rows = df.sample(n=min(5, len(df)))  #pick up to 5 random entries
        #send each word, definition, example sentence and its translation as a separate message
        for idx, row in random_rows.iterrows():
            word = row.iloc[WORD_COLUMN]
            definition = row.iloc[DEFINITION_COLUMN]
            example_sentence = row.iloc[EXAMPLE_SENTENCE_COLUMN]
            translation = row.iloc[TRANSLATION_COLUMN]
            await context.bot.send_message(chat_id=context.job.chat_id, text=f"Word: {word}\nDefinition: {definition}\nExample Sentence: {example_sentence}\nTranslation: {translation}")
    except Exception as e:
        logging.error(f"Error in anki_integration: {e}", exc_info=True)
        try:
            await context.bot.send_message(chat_id=context.job.chat_id, text="Sorry, I encountered an error while trying to send your Anki words.")
        except Exception:
            pass  #ignore failures to send the error message, like if the user has blocked the bot
        return

if __name__ == "__main__":
    #validate required environment variables up front so startup fails with a clear message, not a cryptic traceback
    if not TELEGRAM_TOKEN:
        raise SystemExit("TELEGRAM_TOKEN is missing. Add it to your .env file.")
    if not OWNER:
        raise SystemExit("OWNER is missing. Add your numeric Telegram user ID to your .env file.")
    try:
        owner_id = int(OWNER)
    except ValueError:
        raise SystemExit("OWNER must be a numeric Telegram user ID")

    #log once at startup whether the optional language rules file was found
    if load_language_rules():
        logging.warning("Loaded additional language rules from additional_language_rules.txt")
    else:
        logging.warning("No additional language rules file found, continuing without it")

    #keep the bot alive if polling ever crashes unexpectedly, log it and restart after a short delay
    while True:
        try:
            allowed_users = filters.User(user_id=owner_id)  # restrict the bot to the owner only
            app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
            #add command handlers (filters go inside CommandHandler, not on add_handler)
            app.add_handler(CommandHandler("start", start_command, filters=allowed_users))
            app.add_handler(CommandHandler("profile", profile_command, filters=allowed_users))
            app.add_handler(CommandHandler("anki", anki_command, filters=allowed_users))
            #all incoming text messages that are not commands go to the message handler
            app.add_handler(MessageHandler(allowed_users & filters.TEXT & ~filters.COMMAND, message_command))

            #daily reminder in the early morning when there is least likely to be any traffic
            if app.job_queue is not None:
                reminder_time = datetime.time(hour=REMINDER_HOUR, minute=0, tzinfo=pytz.timezone(TIMEZONE))
                app.job_queue.run_daily(daily_reminder, time=reminder_time, data=owner_id, name="daily_anki")
            else:
                logging.warning("Daily Anki reminder not available.")
            #start the bot
            print("LanguageLearnAI is running...")
            app.run_polling()
            break  #intentional shutdown, exit the loop
        except KeyboardInterrupt:
            print("Bot stopped by user.")
            break
        except Exception as e:
            logging.error(f"Polling crashed, restarting in 5 seconds: {e}", exc_info=True)
            time.sleep(5)