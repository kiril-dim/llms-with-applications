---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
  }
  section.title {
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.title h1 {
    font-size: 2.2em;
  }
  h1 {
    color: #2d3436;
  }
  h2 {
    color: #0984e3;
  }
  table {
    font-size: 0.9em;
  }
---

<!-- _class: title -->

# Големи езикови модели и приложения

## Large language models and applications

Летен семестър 2026
Четвъртък, 17:00–20:00
Хорариум: 2 часа лекции + 1 час упражнение

<https://github.com/kiril-dim/llms-with-applications>

![w:180](./qr_code.jpeg)

---

## Преподаватели

**Кирил Димитров**
10+ години практически опит в ML
Магистър по математика и компютърни науки, Оксфордски университет

**Мелания Бербатова**
Докторант по машинно самообучение, асистент в ФМИ
3+ години практически опит

**Иван Иванов**
ИИ предприемач, магистър от Университета на Бон

---

## Оценяване

| Компонент | Тежест |
|---|---|
| Групов проект | 50% |
| 3 теста (по време на семестъра) | 40% |
| Участие | 10% |

**Минимум за преминаване: 40%**

---

## Предварителни изисквания и инструменти

**Знания**

- Python
- Линейна алгебра, статистика
- Курс по ML е полезен, но не е необходим

**Инструменти**

- Jupyter Notebook, Python scripts
- PyTorch, Transformers, Hugging Face
- Git

---

## Теми на курса

<div style="display: flex; gap: 2em;">
<div>

1. Въведение в ИИ и машинно самообучение
2. От линейни модели до невронни мрежи
3. Токенизация
4. Механизми на вниманието
5. Трансформатор архитектура и дълъг контекст
6. Базови модели и данни за предварително обучение

</div>
<div>

7. Закони за мащабиране на ГЕМ
8. Съгласуване на ИИ и обучение с подсилване чрез обратна връзка
9. Локални езикови модели
10. Инженерство на подканите и разсъждаващи модели
11. Халюцинации и RAG
12. ИИ агенти и инструменти

</div>
</div>

---

## Полезни материали

- **Материали от курса** (GitHub)
  <https://github.com/kiril-dim/llms-with-applications>
- **Speech and Language Processing** (Jurafsky & Martin)
  <https://web.stanford.edu/~jurafsky/slp3/>
- **Stanford CS224N**: NLP with Deep Learning
  <https://web.stanford.edu/class/cs224n/>
- **nanoGPT** (Andrej Karpathy)
  <https://github.com/karpathy/nanoGPT>
- **Neural Networks: Zero to Hero** (Karpathy)
  <https://karpathy.ai/zero-to-hero.html>
