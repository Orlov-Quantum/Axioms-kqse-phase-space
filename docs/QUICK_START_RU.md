# Как загрузить и запустить репозиторий (Windows)

1. Создай на компьютере локальную копию репозитория `Orlov-Quantum/Axioms-kqse-phase-space` через GitHub Desktop (**File → Clone repository**). Существующие `README.md`, `.gitignore` и `LICENSE` останутся на месте.
2. Распакуй **публикационный** ZIP и скопируй его содержимое (папки `notebooks`, `scripts`, `docs`, `results`, `data`, `audit` и файлы README и requirements) в клонированный репозиторий. `README.md` и `.gitignore` разрешено заменить; `LICENSE` в ZIP отсутствует, существующую MIT оставь без изменений.
3. В GitHub Desktop проверь список изменений. **В репозиторий не должны попасть файлы `.dat`, `.npy`, `.npz`, `.pkl`, чужие ноутбуки и приватные ZIP.**
4. Сначала создай коммит локально (`Summary: Add reproducible KQSE numerical studies`), затем нажми **Push origin**. Репозиторий пока оставь Private.
5. Для установки на Windows открой PowerShell в папке репозитория и выполни `python -m pip install -r requirements.txt`.
6. Запусти `python -m jupyter lab` из той же папки. Начни с `notebooks/01_small_cat_and_scaling.ipynb` или прочти сохранённые таблицы `results/tables/` (полный запуск 100 повторов длительный).
7. Для большого кота и экспериментального kitten локально импортируй собственные ZIP с исходными данными командой `python scripts/prepare_local_data.py --large-cat "...zip" --lvovsky "...zip" --physicality "...zip"` (имена укажи свои). После этого `python scripts/check_local_inputs.py` проверит наличие файлов. Эти данные игнорируются Git.
8. Перед переключением на Public проверь источники внешних результатов и согласуй содержание с соавторами. README и научную статью можно привести к окончательной формулировке позднее.

Научные вычисления мы не переписывали: только изменили настройки путей в двух первых ячейках и удалили сохранённые выводы из копий ноутбуков. Таблицы и рисунки приложены отдельно для просмотра.
