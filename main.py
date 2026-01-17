import logging
import os
import discord
import torch
import lightning as L
from discord.ext import commands
from dotenv import load_dotenv
from model import BiLSTMClassifier
from transformers import AutoTokenizer

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

model = BiLSTMClassifier.load_from_checkpoint("bilstm_final.ckpt")
model.eval()

tokenizer_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user.name}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    inputs = tokenizer(
        message.content,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    input_ids = inputs["input_ids"]
    embedding_layer = torch.nn.Embedding(num_embeddings=tokenizer.vocab_size, embedding_dim=150)
    embedded_inputs = embedding_layer(input_ids)

    with torch.no_grad():
        outputs = model(embedded_inputs)

    if outputs[0][0] > -5.0:
        await message.reply(f"Wylkryto mowę nienawiści! | {outputs}")
    else:
        await message.reply(f"Wszystko dobrze. | {outputs}")

    await bot.process_commands(message)

bot.run(TOKEN, log_handler=handler, log_level=logging.DEBUG)
