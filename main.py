import asyncio
import functools
import logging
import os
import discord
import torch
import torch.nn.functional as F
from discord.ext import commands
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

model_id = "model/deberta_best_model"
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    use_safetensors=True,
    num_labels=2,
)
model.to(device)
model.eval()

# tokenizer_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    fix_mistral_regex=True,
)

def predict_async(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)

    hate_speech_prob = probs[0][1].item()

    predicted_label = torch.argmax(logits, dim=-1).item()
    confidence = probs[0, predicted_label].item()

    return hate_speech_prob, confidence

@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user.name}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    THRESHOLD = 0.75

    loop = asyncio.get_event_loop()
    hate_speech_prob, confidence = await loop.run_in_executor(
        None,
        functools.partial(predict_async, message.content),
    )

    if hate_speech_prob > THRESHOLD:
        await message.reply(f"Wykryto mowę nienawiści! ({confidence * 100:.2f})")
    else:
        await message.reply(f"Wszystko dobrze. ({confidence * 100:.2f})")

    await bot.process_commands(message)

bot.run(TOKEN, log_handler=handler, log_level=logging.DEBUG)
