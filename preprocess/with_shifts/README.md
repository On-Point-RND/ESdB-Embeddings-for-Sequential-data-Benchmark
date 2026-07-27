
# Предобработка датасета

## Запуск препроцессинга

Команды необходимо запускать из корня проекта.

Общий шаблон запуска:

```bash
python -m preprocess.with_shifts.<dataset> \
    --data-path data/<dataset>/data \
    --save-path data/<dataset>/preprocessed/ \
    --cat-codes-path data/<dataset>/preprocessed/cat_codes \
    --overwrite
```

Чтобы запрепроцессить полный датасет:

```bash
python -m preprocess.with_shifts.<dataset> \
    --data-path data/<dataset>/data \
    --save-path data/<dataset>/preprocessed_full/ \
    --cat-codes-path data/<dataset>/preprocessed_full/cat_codes \
    --overwrite \
    --user-sample-frac 1
```

Вместо `<dataset>` необходимо указать название модуля препроцессинга
датасета. Параметр `--user-sample-frac 1` включает в выборку всех
пользователей.

Доступные значения `<dataset>`:

- `30music`
- `age`
- `alpha`
- `electric_devices`
- `ett`
- `favorita`
- `rossman`
- `taobao`
- `twitter`
- `x5-retail`
- `yambda`
- `zvuk`

Параметр `--user-sample-frac` поддерживают `30music`, `age`, `alpha`,
`twitter`, `x5-retail`, `yambda` и `zvuk`. Остальные модули обрабатывают
весь переданный датасет без этого параметра.

## Формат именования таргетов

При определении целевых переменных (таргетов) необходимо соблюдать следующий формат именования:

```
target__<name>__<local/global>__<metric1+metric2+...>
```

### Пояснение полей:
- **`<name>`** — краткое название таргета (например, `reg_amount`, `age`, `anomaly`);
- **`<local/global>`** — указывает, является ли задача локальной (на уровне отдельного события или сессии) или глобальной (на уровне всей последовательности);
- **`<metric1+metric2+...>`** — метрики, по которым будет оцениваться качество модели (перечисляются через `+` без пробелов).

### Примеры корректных имён:
- `target__reg_amount__local__mse+r2`
- `target__age__global__accuracy+f1_macro`
- `target__anomaly__local__roc_auc+f1_macro+accuracy`
