import telebot
from telebot import types
import os
from dotenv import load_dotenv
from processing import (
    extract_text_from_txt,
    extract_text_from_pdf,
    extract_text_from_docx,
    split_into_chunks,
    embed_chunks,
    #create_faiss_index,
    #find_relevant_chunks,
    generate_response,
    check_gpu_support
)

load_dotenv('token.env')
token = os.getenv("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(token=token)

# === НАСТРОЙКИ ===
USE_LLM = False   # <-- переключи на False чтобы отключить LLM

# === ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ===
all_texts = []
saved_requests = []
text_chunks = []
faiss_index = None
gpu_available = check_gpu_support() if USE_LLM else False

@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(
        types.KeyboardButton("Загрузить файл"),
        types.KeyboardButton("📝 Задать вопрос")
    )
    bot.send_message(message.chat.id, "👋 Привет! Пришли файл, затем задай вопрос.", reply_markup=markup)

@bot.message_handler(content_types=['text', 'document'])
def handle_message(message):
    global all_texts, text_chunks, faiss_index

    if message.document:
        file_info = bot.get_file(message.document.file_id)
        file_bytes = bot.download_file(file_info.file_path)
        name = message.document.file_name.lower()

        if name.endswith(".txt"):
            text = extract_text_from_txt(file_bytes)
        elif name.endswith(".pdf"):
            text = extract_text_from_pdf(file_bytes)
        elif name.endswith(".docx"):
            text = extract_text_from_docx(file_bytes)
        else:
            bot.send_message(message.chat.id, "❌ Поддерживаются только TXT, PDF, DOCX")
            return

        all_texts.append(text)
        faiss_index = None  # пересоздание при новых файлах
        bot.send_message(message.chat.id, f"✅ Файл загружен: {name}, слов: {len(text.split())}")

    elif message.text == "📝 Задать вопрос":
        bot.send_message(message.chat.id, "✍️ Введите ваш вопрос:")
        bot.register_next_step_handler(message, process_query)

    elif message.text:
        # также обрабатываем просто текстовое сообщение как вопрос
        process_query(message)

def process_query(message):
    global faiss_index, text_chunks
    query = message.text
    saved_requests.append(query)

    if not all_texts:
        bot.send_message(message.chat.id, "❌ Сначала загрузите документы")
        return

    if faiss_index is None:
        bot.send_message(message.chat.id, "⏳ Строим индекс...")
        text_chunks.clear()
        for doc in all_texts:
            text_chunks.extend(split_into_chunks(doc))
        embeddings = embed_chunks(text_chunks)
        #faiss_index = create_faiss_index(embeddings)

    bot.send_message(message.chat.id, "🔍 Ищем релевантные фрагменты...")
    relevant = True #find_relevant_chunks(query, faiss_index, text_chunks)

    if not relevant:
        bot.send_message(message.chat.id, "❌ Не найдено релевантных фрагментов")
        return

    context = "\n\n".join(relevant)

    if USE_LLM:
        bot.send_message(message.chat.id, "🧠 Генерируем ответ с помощью LLM...")
        answer = generate_response(query, context, gpu_available)
    else:
        answer = f"📚 Найдено {len(relevant)} фрагментов:\n\n{context}"

    for i in range(0, len(answer), 4000):
        bot.send_message(message.chat.id, answer[i:i+4000])

bot.polling(none_stop=True)
