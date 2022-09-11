# Tinkoff N-gram

## Used external libraries

- `numpy` - for "weighted" random (`numpy.choice` allows to pass probability of taking each item)
- `tqdm` - for pretty progress bars

## Limitations

Deadline is in a few hours and I think I made a mistake: all the NGrams have the same size.
I suppose it's quite late to fix that, but trust me: I can.
They were also generated from all parts of the sentence, not only the beginning.

## Installation

Using poetry:

```
poetry install
poetry run python3 train.py --.... # params here
poetry run python3 generate.py --.... # params here
```

Using pip:

```
pip3 install -r requirements.txt
python3 train.py --.... # params here
generate.py --.... # params here
```

## Results

Both of the results were not moderated, so I am not providing only the best examples.
Both examples used prefix of length 3.

### Random generation (without prefix):

```
Sentence(words=['смотря', 'на', 'темные', 'воды', 'рейна', 'здесь'])
Sentence(words=['сначала', 'я', 'только', 'неохотно', 'подавал', 'реплики'])
Sentence(words=['их', 'не', 'обнаруживая', 'интереса', 'к', 'тому'])
Sentence(words=['розово-красными', 'сережками', 'сергей', 'иванович', 'зная', 'что'])
Sentence(words=['наши', 'одноклассники', 'лохи', '-', 'до', 'сих'])
```

And longer ones:

```
Sentence(words=['пары-сказать', 'что', 'я', 'в', 'прошлый', 'раз', 'семен', 'яковлевич', 'спросил', 'он', 'вдову', 'тихим', 'и', 'размеренным', 'голосом'])
Sentence(words=['по', 'высшим', 'приказаниям', 'вот', 'он', 'дал', 'бумагу', 'алпатычу'])
Sentence(words=['сформировавшемуся', 'в', 'этот', 'девятилетний', 'срок', 'разлуки', 'наталья', 'васильевна', 'принадлежала', 'к', 'числу', 'матушек-командирш', 'носила', 'пышные', 'чепцы'])
Sentence(words=['утро', 'доброе', 'жуйк', 'вот', 'вам', 'пример', 'вместо', 'того', 'чтоб', 'овладеть', 'свободой', 'людей', 'ты', 'увеличил', 'им'])
Sentence(words=['с', 'помощью', 'протокола', 'обмена', 'что-то', 'меня', 'на', 'арахис', 'пробило', 'купил', 'две', 'банки', 'по', 'гр', 'одну'])
```

### Generation with prefix:

```
Sentence(words=['интереса', 'к', 'тому', 'о', 'чем', 'говорили'])
Sentence(words=['интереса', 'к', 'тому', 'о', 'чем', 'объявили'])
Sentence(words=['интереса', 'к', 'тому', 'о', 'чем', 'замышляют'])
Sentence(words=['интереса', 'к', 'тому', 'о', 'чем', 'она'])
Sentence(words=['интереса', 'к', 'тому', 'о', 'чем', 'вы'])
```

And longer ones:
```
Sentence(words=['интереса', 'к', 'тому', 'о', 'чем', 'ты', 'подумала', 'наверняка', 'буквально', 'в', 'пятницу', 'уходил', 'с', 'работы', 'или'])
Sentence(words=['интереса', 'к', 'тому', 'о', 'чем', 'говорили', 'они', 'это', 'делало', 'меня', 'счастливым', 'жизнь', 'мою', 'легкой', 'и'])
Sentence(words=['интереса', 'к', 'тому', 'о', 'чем', 'вы', 'теперь', 'думаете', 'вы', 'думаете', 'что', 'с', 'ними', 'делать', '--'])
```

## Sources of data

- [Russian Literature](https://www.kaggle.com/datasets/d0rj3228/russian-literature) on Kaggle
- [RuTweetCorp](https://study.mokoron.com/#download)

## Citations

```
Рубцова Ю. Автоматическое построение и анализ корпуса коротких текстов (постов микроблогов) для задачи разработки и тренировки тонового классификатора //Инженерия знаний и технологии семантического веба. – 2012. – Т. 1. – С. 109-116.
```
