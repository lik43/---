from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
import asyncio
import requests

token = '7425199798:AAGCzBf6oygoadUY9CVhD5kB0NJEpgLMCPQ'
bot = Bot(token)
dp= Dispatcher()

kb_load = [
    [types.KeyboardButton(text='/load')],
    [types.KeyboardButton(text='/list_models')]
]
keyboard_load = types.ReplyKeyboardMarkup(keyboard=kb_load, resize_keyboard=True)

kb_process = [[types.KeyboardButton(text='/process')]]
keyboard_process = types.ReplyKeyboardMarkup(keyboard=kb_process, resize_keyboard=True)

kb_predict = [[types.KeyboardButton(text='/predict')]]
keyboard_predict = types.ReplyKeyboardMarkup(keyboard=kb_predict, resize_keyboard=True)

kb_choose_models = [[types.KeyboardButton(text='/choose_models')]]
keyboard_choose_models = types.ReplyKeyboardMarkup(keyboard=kb_choose_models, resize_keyboard=True)



@dp.message(Command("start"))
async def cmd_start(msg: types.Message):
    await msg.answer("Привет, эксперт!", reply_markup=keyboard_load)

@dp.message(Command('load'))
async def load(msg:types.Message):
    await msg.answer('Файлы, загружены', reply_markup=keyboard_process)

@dp.message(Command('predict'))
async def predict(msg:types.Message):
    r = requests.get('http://127.0.0.1:8000/predict')
    text = r.json().get('y_pred')
    await msg.answer(str(text), reply_markup=keyboard_load)

@dp.message(Command('list_models'))
async def list_models(msg:types.Message):
    list_model = requests.get('http://127.0.0.1:8000/list_models').json()
    await msg.answer(','.join(list_model), reply_markup=keyboard_choose_models)


@dp.message(Command('choose_models'))
async def choose_models(msg:types.Message):
    ch_model = requests.get('http://127.0.0.1:8000/choose_model').text
    await msg.answer(ch_model, reply_markup=keyboard_load)

@dp.message(Command('process'))
async def choose_models(msg:types.Message):
    await msg.answer('Файлы обработаны', reply_markup=keyboard_predict)

# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())