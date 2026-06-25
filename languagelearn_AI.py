import os
import logging
import json
from dotenv import load_dotenv
from telegram import Update
from telegram.error import Forbidden, BadRequest
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, filters, CallbackQueryHandler, MessageHandler
from google import genai
from google.genai import types
import asyncio
import time

#resources
#https://ai.google.dev/gemini-api/docs

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
#use utf-8 encoding to support non-ASCII characters in the chat history
def load_history(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        #only accept a list; anything else like corrupted JSON is treated as empty history
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []

def save_history(history, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
    except OSError as e:
        logging.error(f"Failed to save history to {file_path}: {e}")

#load the user's plain-text profile, or an empty string if none has been saved yet
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
    user_profile = "The following information was provided about the user:\n\n" + message
    #save the user's profile information to a file
    try:
        with open("user_profile.txt", "w", encoding="utf-8") as f:
            f.write(user_profile)
    except OSError as e:
        logging.error(f"Failed to save profile: {e}")
        await update.message.reply_text("Sorry, I couldn't save your profile right now. Please try again.")
        return

    await update.message.reply_text("Thanks! Your profile information has been saved. I will use this information to tailor our conversations to your needs.")

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
            types.Part.from_text(text=f"""You are a friendly, supportive male {BOT_LANGUAGE} friend to the user. Your primary goal is to maintain a natural, engaging conversation while gently aiding the user's Japanese learning journey.

Core Persona:
- Your name is {BOT_NAME}.
- You speak casually but grammatically correctly (e.g., standard plain form / dictionary form, casual polite forms when appropriate for a friend).
- You are playful, warm, and have a good sense of humor. You text like a real friend chatting on LINE: expressive, relaxed, and fun, while still sounding like a normal, well-educated {BOT_AGE}-year-old university student in {BOT_LOCATION}. Avoid heavy anime or drama tropes.
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
- Keep your messages relatively short, typical of a Telegram text message (1-3 sentences).
- If you must explain a concept in English, clearly separate it from the conversational {BOT_LANGUAGE}."""),
    ]
    #add the additional language and personality rules if present
    language_rules = load_language_rules()
    if language_rules:
        system_instruction.append(types.Part.from_text(text=language_rules))
    #read the user's saved profile and add it to the system instruction if present
    user_profile = load_profile()
    if user_profile:
        system_instruction.append(types.Part.from_text(text=user_profile))

    generate_content_config = types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=500,
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
        chat = client.chats.create(
            model=model,
            config=generate_content_config,
            history=history
        )
        #send the user's message to the model and get the response
        response = await asyncio.to_thread(chat.send_message, message)
        #send the telegram response
        await update.message.reply_text(response.text or "Sorry, I couldn't generate a response at this time.")
        #append this turn to the history and persist it
        if response.text:
            records.append({"role": "user", "text": message})
            records.append({"role": "model", "text": response.text})
            if len(records) > 80:  # Keep only the last 80 messages for context
                records = records[-80:]
            save_history(records, "history.json")

            print(f"User: {message}")
            print(f"{BOT_NAME}: {response.text}") #if failed response.text is blank

    except Exception as e:
        logging.error(f"Error in message_command: {e}", exc_info=True)
        try:
            await update.message.reply_text("Sorry, I encountered an error while processing your request.")
        except Exception:
            pass  #ignore failures to send the error message (e.g. the user blocked the bot)
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

    #keep the bot alive if polling ever crashes unexpectedly, log it and restart after a short delay
    while True:
        try:
            allowed_users = filters.User(user_id=owner_id)  # restrict the bot to the owner only
            app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
            #add command handlers (filters go inside CommandHandler, not on add_handler)
            app.add_handler(CommandHandler("start", start_command, filters=allowed_users))
            app.add_handler(CommandHandler("profile", profile_command, filters=allowed_users))
            #all incoming non-command text messages go to the message handler
            app.add_handler(MessageHandler(allowed_users & filters.TEXT & ~filters.COMMAND, message_command))

            #start the bot
            print("LanguageLearnAI is running...")
            app.run_polling()
            break  #run_polling returned normally (e.g. Ctrl+C) -> intentional shutdown, don't restart
        except KeyboardInterrupt:
            print("Bot stopped by user.")
            break
        except Exception as e:
            logging.error(f"Polling crashed, restarting in 5 seconds: {e}", exc_info=True)
            time.sleep(5)