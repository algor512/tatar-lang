## Подготовка данных

Надо положить скрипты flattenize.sh и tag2vec.sh на одну папку ниже 
файлов words.csv и disamed.csv (например, в подпапку scripts/). 
Далее запустить:

```bash
./flattenize.sh
./vec2tag.sh
./create_dataset.py all_tags.txt vectorized.tsv ans_vectorized.tsv sentences.jsonl # первые три файла - результат работы предыдущих скриптов
```

Теперь в sentences.jsonl записан необходимый для обучения и тестирования датасет.

## Обучение и тестирование

Обучать модель можно, например, так:
```bash
./model_creator.py sentences.jsonl model.json roots.txt --texts 1 30 --states 10
```

Тестировать:
```bash
./model_tester_new.py sentences.jsonl model.json results.jsonl
```
На выходе --- jsonl-файл с результатами тестирования по предложениям.
