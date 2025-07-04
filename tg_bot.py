
import telebot
from telebot import types  # Для создания клавиатуры и кнопок
import io  # Для работы с байтовыми потоками в памяти (BytesIO)
from PyPDF2 import PdfReader  # Для извлечения текста из PDF файлов
from sentence_transformers import SentenceTransformer  # Для создания эмбеддингов текста

# Загружаем предобученную модель для эмбеддингов
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

token = "token-bot"  # Токен вашего Telegram-бота
bot = telebot.TeleBot(token=token)  # Инициализация бота с токеном


# Глобальный список для хранения всех загруженных текстов (из файлов) и запросов
all_texts = []
saved_requests = []

# Функция для извлечения текста из .txt файла (байты -> строка)
def extract_text_from_txt(file_bytes: bytes) -> str:
    # Декодируем байты в строку UTF-8 (если кириллица, можно сменить на 'cp1251')
    return file_bytes.decode('utf-8')

# Функция для извлечения текста из PDF файла (байты -> строка)
def extract_text_from_pdf(file_bytes: bytes) -> str:
    # Открываем PDF из байтового потока без сохранения на диск
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ''
    # Проходим по всем страницам PDF и собираем текст
    for page in reader.pages:
        text += page.extract_text() or ''  # добавляем текст страницы, если есть
    return text

def save_request(message):
    global save_request
    saved_requests.append(message.text)
    print(saved_requests)
    bot.send_message(message.chat.id, f"✅ Запрос сохранён:\n\n{message.text}")

# Функция для создания эмбеддинга текста
def embed_text(text: str):
    # Модель принимает список текстов, поэтому передаем список из одного элемента
    return embedding_model.encode([text])

# Функция для разбивки длинного текста на чанки (части) по max_words слов
def split_into_chunks(text: str, max_words: int = 500):
    words = text.split()  # Разбиваем текст на отдельные слова
    # Формируем список строк-чанков по max_words слов в каждом
    chunks = [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return chunks


# Обработчик команды /start — приветственное сообщение и кнопка
@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)  # Создаем клавиатуру
    btn1 = types.KeyboardButton("👋 И тебе не хворать")  # Кнопка с текстом
    markup.add(btn1)
    bot.send_message(message.chat.id, "👋 Привет!", reply_markup=markup)  # Отправляем сообщение с клавиатурой


# Основной обработчик сообщений с текстом и файлами
@bot.message_handler(content_types=['text', 'document'])
def handle_files(message):
    global all_texts  # Объявляем, что будем использовать глобальную переменную для списка текстов

    if message.document:
        # Получаем информацию о файле и его содержимое в байтах
        file_info = bot.get_file(message.document.file_id)
        file_name = message.document.file_name
        file_bytes = bot.download_file(file_info.file_path)

        try:
            # Определяем расширение файла и выбираем функцию для извлечения текста
            if file_name.split('.')[-1].lower() == 'txt':
                text = extract_text_from_txt(file_bytes)
            elif file_name.split('.')[-1].lower() == 'pdf':
                text = extract_text_from_pdf(file_bytes)
            else:
                bot.reply_to(message, '❌ Поддерживаются только файлы .txt и .pdf')
                return

            all_texts.append(text)  # Добавляем извлеченный текст в глобальный список
            print(len(all_texts))  # Логируем количество загруженных файлов
            bot.reply_to(message, f"📄 Получен файл: {message.document.file_name}")

        except Exception as er:
            # Если произошла ошибка при обработке файла — уведомляем пользователя
            bot.reply_to(message, f"⚠️ Ошибка при обработке файла: {er}")

    elif message.text:
        # Обработка текстовых команд от пользователя
        if message.text == '👋 И тебе не хворать':
            # Создаем клавиатуру с кнопками для управления файлами
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            btn1 = types.KeyboardButton("Добавить файл")
            btn2 = types.KeyboardButton('Загрузить файлы')
            btn3 = types.KeyboardButton('Удалить все файлы и запросы')
            btn4 = types.KeyboardButton("📝 Написать запрос")
            markup.add(btn1)
            markup.add(btn2)
            markup.add(btn3)
            markup.add(btn4)
            bot.send_message(message.chat.id, 'Начнем?)', reply_markup=markup)

        elif message.text == 'Добавить файл':
            bot.send_message(message.chat.id, 'Присылайте файл')

        elif message.text == 'Загрузить файлы':
            # Разбиваем все загруженные тексты на чанки
            all_chunks = []
            for doc in all_texts:
                chunks = split_into_chunks(doc, max_words=500)
                all_chunks.extend(chunks)

            print(f'Created {len(all_chunks)} chunks')  # Логируем количество чанков
            # Здесь можно добавить вызов embed_text(all_chunks) для создания эмбеддингов
           

        elif message.text == 'Удалить все файлы и запросы':
            all_texts = []  # Очищаем список загруженных текстов
            saved_requests = [] # Очистка запросов
            print(saved_requests)
            bot.send_message(message.chat.id, 'Файлы и запросы удалены')
            

        elif message.text == '📝 Написать запрос':
            bot.send_message(message.chat.id, "Напиши свой запрос:")
            bot.register_next_step_handler(message, save_request)
            
            
            
# Запуск бота — постоянное прослушивание сообщений
bot.polling(none_stop=True, interval=0)
