import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import time
import json
from datetime import datetime
import os

# Настройка страницы
st.set_page_config(
    page_title="Агент оценки звонков",
    page_icon="📞", 
    layout="wide"
)

tok = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# Кэширование моделей для ускорения работы
@st.cache_resource
def load_sentiment_model():
    """Загружает модель анализа тональности с усечением длинного текста"""
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device,
        return_all_scores=True,
        tokenizer=model_name,
        truncation=True,      # обрезать длинные тексты
        max_length=512         # максимум 512 токенов
    )

@st.cache_resource 
def load_generation_model():
    """Загружает модель для генерации рекомендаций"""
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = model.to('cuda')

    return tokenizer, model

def preprocess_text(text):
    """Предобработка текста для анализа"""
    # Замена упоминаний пользователей и ссылок
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def chunk_and_analyze(text, pipe, tokenizer, max_len=512, stride=448):
    """Разбивает текст на перекрывающиеся чанки, анализирует каждый и агрегирует."""
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    results = []
    for start in range(0, len(token_ids), stride):
        chunk_ids = token_ids[start : start + max_len]
        chunk = tokenizer.decode(chunk_ids, clean_up_tokenization_spaces=True)
        # pipe(chunk) возвращает [[{label,score},…]]
        chunk_scores = pipe(chunk)[0]  
        # для этого чанка выбираем словарь с максимальным score
        main_chunk = max(chunk_scores, key=lambda x: x["score"])
        results.append(main_chunk)
    # агрегируем: здесь выбираем самый уверенный по всем чанкам
    best = max(results, key=lambda x: x["score"])
    return best, results

def analyze_sentiment(text, model):
    processed = preprocess_text(text)
    input_ids = tok(processed)["input_ids"]
    if len(input_ids) <= 512:
        scores = model(processed)[0]  # list of dicts
        main = max(scores, key=lambda x: x["score"])
        return main, scores
    else:
        return chunk_and_analyze(processed, model, tok)

def generate_recommendations(sentiment, text):
    """Генерирует рекомендации на основе анализа"""
    recommendations = []

    if sentiment['label'] == 'NEGATIVE':
        recommendations = [
            "🤝 Проявите больше эмпатии к проблеме клиента",
            "✅ Предложите конкретные шаги для решения проблемы",
            "🎯 Уточните детали проблемы для лучшего понимания",
            "📞 Предложите дополнительную консультацию"
        ]
    elif sentiment['label'] == 'NEUTRAL':
        recommendations = [
            "😊 Добавьте больше позитива в общение",
            "❓ Задайте уточняющие вопросы о потребностях",
            "🎪 Сделайте разговор более вовлекающим",
            "💡 Предложите дополнительные возможности"
        ]
    else:  # POSITIVE
        recommendations = [
            "🌟 Поддержите позитивный настрой клиента",
            "🎁 Предложите дополнительные услуги или продукты",
            "📈 Узнайте о возможности расширения сотрудничества",
            "💬 Попросите оставить отзыв или рекомендацию"
        ]

    return recommendations[:2]  # Возвращаем только 2 рекомендации

def save_analysis_history(text, sentiment, recommendations):
    """Сохраняет историю анализа"""
    if not os.path.exists('history'):
        os.makedirs('history')

    history_entry = {
        'timestamp': datetime.now().isoformat(),
        'text': text[:100] + "..." if len(text) > 100 else text,
        'sentiment': sentiment['label'],
        'confidence': sentiment['score'],
        'recommendations': recommendations
    }

    history_file = 'history/analysis_history.json'

    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []

    history.append(history_entry)

    # Оставляем только последние 100 записей
    history = history[-100:]

    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# Заголовок приложения
st.title("📞 Агент оценки звонков")
st.markdown("**Анализ тональности разговоров и генерация рекомендаций**")

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Настройки")

    # Информация о системе
    device_info = "🖥️ CPU" if not torch.cuda.is_available() else f"🚀 GPU ({torch.cuda.get_device_name(0)})"
    st.info(f"Устройство: {device_info}")

    # Статистика моделей
    st.header("📊 Модели")
    st.write("**Анализ тональности:**")
    st.code("cardiffnlp/twitter-roberta-base-sentiment", language="text")

    # История анализов
    if st.button("📈 Показать историю"):
        try:
            with open('history/analysis_history.json', 'r', encoding='utf-8') as f:
                history = json.load(f)

            if history:
                st.write(f"**Всего анализов:** {len(history)}")
                for entry in history[-5:]:  # Показываем последние 5
                    sentiment_emoji = {"POSITIVE": "😊", "NEGATIVE": "😢", "NEUTRAL": "😐"}
                    st.write(f"{sentiment_emoji.get(entry['sentiment'], '🤔')} {entry['sentiment']} ({entry['confidence']:.1%})")
        except FileNotFoundError:
            st.write("История пуста")

# Загрузка моделей с индикатором прогресса
if 'sentiment_model' not in st.session_state:
    with st.spinner('🔄 Загрузка модели анализа тональности...'):
        st.session_state.sentiment_model = load_sentiment_model()
    st.success('✅ Модель загружена!')

# Основной интерфейс
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Ввод данных")

    # Текстовое поле для ввода
    call_text = st.text_area(
        "Введите расшифровку разговора:",
        height=200,
        placeholder="Оператор: Здравствуйте! Это служба поддержки...\nКлиент: Привет, у меня проблема с заказом..."
    )

    # Кнопка для примера
    if st.button("📋 Использовать пример"):
        example_text = """Оператор: Здравствуйте! Это служба поддержки. Как дела?
Клиент: Привет. У меня проблема с заказом. Я жду уже неделю, а товар не пришел.
Оператор: Понимаю ваше беспокойство. Давайте разберемся. Назовите номер заказа.
Клиент: 12345. Я очень расстроен этой ситуацией.
Оператор: Я проверю статус заказа и решу проблему в кратчайшие сроки."""
        st.session_state.example_text = example_text
        st.experimental_rerun()

    if 'example_text' in st.session_state:
        call_text = st.session_state.example_text
        del st.session_state.example_text

with col2:
    st.header("⚡ Действия")

    analyze_button = st.button("🔍 Анализировать разговор", type="primary")

    if st.button("🗑️ Очистить"):
        st.experimental_rerun()

# Анализ текста
if analyze_button and call_text:
    with st.spinner('🔄 Анализируем...'):
        start_time = time.time()

        # Анализ тональности
        main_sentiment, all_sentiments = analyze_sentiment(call_text, st.session_state.sentiment_model)

        # Генерация рекомендаций
        recommendations = generate_recommendations(main_sentiment, call_text)

        # Сохранение в историю
        save_analysis_history(call_text, main_sentiment, recommendations)

        end_time = time.time()
        processing_time = end_time - start_time

    # Отображение результатов
    st.header("📊 Результаты анализа")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Основная тональность
        sentiment_colors = {
            'POSITIVE': 'green',
            'NEGATIVE': 'red',
            'NEUTRAL': 'orange'
        }

        sentiment_emojis = {
            'POSITIVE': '😊',
            'NEGATIVE': '😢', 
            'NEUTRAL': '😐'
        }

        color = sentiment_colors.get(main_sentiment['label'], 'gray')
        emoji = sentiment_emojis.get(main_sentiment['label'], '🤔')

        st.metric(
            "Основная тональность",
            f"{emoji} {main_sentiment['label']}",
            f"{main_sentiment['score']:.1%}"
        )

    with col2:
        # Уверенность модели
        confidence_color = "green" if main_sentiment['score'] > 0.8 else "orange" if main_sentiment['score'] > 0.6 else "red"
        st.metric(
            "Уверенность модели", 
            f"{main_sentiment['score']:.1%}",
            "Высокая" if main_sentiment['score'] > 0.8 else "Средняя" if main_sentiment['score'] > 0.6 else "Низкая"
        )

    with col3:
        # Время обработки
        st.metric("Время обработки", f"{processing_time:.2f} сек")

    # Детальная разбивка по всем тональностям
    st.header("📈 Детальный анализ")

    sentiment_df = {
        'Тональность': [],
        'Вероятность': [],
        'Процент': []
    }

    for sentiment in all_sentiments:
        sentiment_df['Тональность'].append(f"{sentiment_emojis.get(sentiment['label'], '🤔')} {sentiment['label']}")
        sentiment_df['Вероятность'].append(sentiment['score'])
        sentiment_df['Процент'].append(f"{sentiment['score']:.1%}")

    st.dataframe(sentiment_df, use_container_width=True)

    # Рекомендации
    st.header("💡 Рекомендации")

    for i, rec in enumerate(recommendations, 1):
        st.write(f"**{i}.** {rec}")

elif analyze_button and not call_text:
    st.warning("⚠️ Пожалуйста, введите текст для анализа")

# Футер
st.markdown("---")
st.markdown("🚀 **Локальный агент анализа звонков** • Работает полностью оффлайн • Модели: RoBERTa + DistilBERT")
