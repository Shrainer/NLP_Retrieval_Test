"""
Pipeline для генерации кандидатов
"""

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import pymorphy3

from cache_utils import (
    items_fingerprint,
    load_or_compute,
    train_fingerprint,
)

_morph = pymorphy3.MorphAnalyzer()


DATA_DIR = os.environ.get(
    "DATA_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)

TRAIN_PATH = os.path.join(
    DATA_DIR,
    "train.parquet",
)

QUERIES_PATH = os.path.join(
    DATA_DIR,
    "benchmark_queries.parquet",
)

ITEMS_PATH = os.path.join(
    DATA_DIR,
    "benchmark_items.parquet",
)

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "answer.csv",
)


# --------------------------------------------------------------------------
# Конфигурация Retrieval
# --------------------------------------------------------------------------

# Сколько возвращаем кандидатов
TOP_K = 50

BM25_FETCH_K = 3000

# Максимальное количество объектов из каждого источника для RRF.
MAX_SOURCE_LEN = 1000

# Используемая Dense-модель
DENSE_MODEL_NAME = "intfloat/multilingual-e5-small"

DENSE_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"

DENSE_CACHE_PATH = os.path.join(
    DATA_DIR,
    "item_emb_e5small_v2_structured.npy",
)

DENSE_DIM = 384
DENSE_FETCH_K = 3000
DENSE_MAX_WORDS = 40
DENSE_MAX_TOKENS = 64
DENSE_BATCH_SIZE = 512
DENSE_SAVE_FP16 = True


# Повторение заголовка в полном тексте для индекса BM25
TITLE_BOOST = 3


# --------------------------------------------------------------------------
# Конфигурация microcat_id
# --------------------------------------------------------------------------

MIN_MICROCAT_POOL = 100
MAX_MICROCAT_POOL = 40000

# Максимальный размер пула для независимого индекса BM25.
MICROCAT_BM25_MAX_POOL = 5000


# --------------------------------------------------------------------------
# Конфигурация RRF
# --------------------------------------------------------------------------

# Режим 1:
# Запрос и локация были в train.
RRF_SEEN_K = 100

W_SEEN_RULE_FINE = 0.20
W_SEEN_RULE_COARSE = 0.20

W_SEEN_BM25_LOCAL = 1.00
W_SEEN_BM25_GLOBAL = 0.05

W_SEEN_TITLE_LOCAL = 0.20
W_SEEN_TITLE_GLOBAL = 0.05

W_SEEN_SERVICE_EXACT = 0.15

W_SEEN_QUERY2QUERY = 0.75
W_SEEN_DENSE = 0.30

W_SEEN_MICROCAT = 0.05


# Режим 2:
# Для всех остальных случаев (примерно 93%)
RRF_GENERAL_K = 60

W_GENERAL_RULE_FINE = 0.00
W_GENERAL_RULE_COARSE = 0.00

W_GENERAL_BM25_LOCAL = 1.50
W_GENERAL_BM25_GLOBAL = 0.15

W_GENERAL_TITLE_LOCAL = 0.20
W_GENERAL_TITLE_GLOBAL = 0.08

W_GENERAL_SERVICE_EXACT = 0.20

W_GENERAL_QUERY2QUERY = 0.80
W_GENERAL_DENSE = 0.25

W_GENERAL_MICROCAT = 0.00

TOKEN_RE = re.compile(
    r"[а-яёa-z0-9]+",
    re.IGNORECASE,
)

# Слова, которые убираются из params из-за неинформативности
_ITEM_MARKERS = [
    "Тип услуги автосервиса",
    "Тип услуги",
    "Вид услуги",
    "Место оказания услуг",
    "Место работы",
    "График работы",
    "Время работы",
    "Работа по договору",
    "Гарантия на работу",
    "Опыт работы",
    "Работа с",
    "Онлайн-запись",
    "Рейтинг пользователя",
    "Кто оказывает услуги",
    "Где вы оказываете услуги",
    "Как вы работаете",
    "Дни",
    "Машины",
]

_ITEM_MARKER_RE = re.compile(
    "|".join(
        re.escape(x)
        for x in _ITEM_MARKERS
    )
)

_SERVICE_KEYS = (
    "Вид услуги",
    "Тип услуги",
    "Тип услуги автосервиса",
)


@dataclass
class PipelineContext:
    """Хранит вычисленные индексы и списки"""

    items_df: pd.DataFrame
    item_id_arr: np.ndarray
    item_id_to_idx: dict
    item_rating: np.ndarray
    item_microcat: np.ndarray
    item_location: np.ndarray
    n_items: int

    rule_fine: dict
    rule_coarse: dict

    bm25_retriever: Any
    bm25_title_retriever: Any

    q_retriever: Any
    unique_queries: list
    q2items: dict

    svc_to_microcat: dict
    microcat_items: dict
    service_value_items: dict

    item_embs: np.ndarray = None

    top_k: int = TOP_K
    bm25_fetch_k: int = BM25_FETCH_K
    rrf_k: int = RRF_GENERAL_K

    min_microcat_pool: int = MIN_MICROCAT_POOL
    max_microcat_pool: int = MAX_MICROCAT_POOL
    microcat_bm25_max_pool: int = MICROCAT_BM25_MAX_POOL

    dense_fetch_k: int = DENSE_FETCH_K


# --------------------------------------------------------------------------
# Утилиты
# --------------------------------------------------------------------------

@lru_cache(maxsize=500_000)
def _lemma(word):
    """Кешируемая лемматизация на русский язык"""
    try:
        forms = _morph.normal_forms(word)
        return forms[0] if forms else word
    except (IndexError, TypeError):
        return word


def tokenize(text):
    """
    Токенизация для индексов BM25, 
    сохраняет буквенно-цифровые обозначения
    """
    if not isinstance(text, str):
        return []

    tokens = TOKEN_RE.findall(
        text.lower()
    )

    out = []

    for token in tokens:
        if any(ch.isdigit() for ch in token):
            out.append(token)
        elif len(token) > 2:
            out.append(_lemma(token))

    return out


def norm_query(text):
    """Нормализация поискового запроса"""
    if not isinstance(text, str):
        return ""

    return " ".join(
        text.lower().split()
    )


def _safe_attr(
    q,
    name,
    default=None,
):
    if isinstance(q, pd.Series):
        return q.get(
            name,
            default,
        )

    if isinstance(q, dict):
        return q.get(
            name,
            default,
        )

    return getattr(
        q,
        name,
        default,
    )


# --------------------------------------------------------------------------
# Парсер структурированных параметров
# --------------------------------------------------------------------------

def parse_infm_params(text) -> dict:
    """
    Превращает строки вида Key Value Key Value в {key: value}
    """
    if not isinstance(text, str):
        return {}

    if not text:
        return {}

    matches = list(
        _ITEM_MARKER_RE.finditer(text)
    )

    if not matches:
        return {}

    out = {}

    for i, match in enumerate(matches):
        key = match.group(0)

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        value = text[
            match.end():end
        ].strip()

        if not value:
            continue

        if key in out:
            out[key] = (
                out[key]
                + " | "
                + value
            )
        else:
            out[key] = value

    return out


def _normalize_service_value(
    value: str,
) -> str:
    value = value.strip().lower()

    value = re.sub(
        r"\s*,\s*",
        ", ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value


def extract_service_values(
    text,
) -> list[str]:
    """
    Получает нормализированные значения сервиса из:
        Вид услуги
        Тип услуги
        Тип услуги автосервиса
    """
    parsed = parse_infm_params(
        text
    )

    values = []

    for key in _SERVICE_KEYS:
        value = parsed.get(key)

        if value:
            values.append(
                _normalize_service_value(
                    value
                )
            )

    return values


# --------------------------------------------------------------------------
# Dense retrieval
# --------------------------------------------------------------------------

_dense_model = None


def get_dense_model():
    global _dense_model

    if _dense_model is None:
        from sentence_transformers import SentenceTransformer

        _dense_model = SentenceTransformer(
            DENSE_MODEL_NAME,
            revision=DENSE_MODEL_REVISION,
        )

    return _dense_model


def _dense_text_for_items(items):
    """
    Структурированное представление для dense-поиска.

    Приоритет:
        название
        значения услуги
        полезные операционные параметры
    """
    titles = (
        items["item_title_raw"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    params = (
        items["item_infm_params_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    output = []

    for title, infm in zip(
        titles,
        params,
    ):
        parts = []

        title = title.strip()

        if title:
            parts.append(
                title
            )

        services = extract_service_values(
            infm
        )

        parts.extend(
            services
        )

        parsed = parse_infm_params(
            infm
        )

        for key in (
            "Место оказания услуг",
            "Место работы",
            "Кто оказывает услуги",
        ):
            value = parsed.get(key)

            if value:
                parts.append(
                    f"{key}: {value}"
                )

        text = " ".join(
            x
            for x in parts
            if x
        )

        words = text.split()

        output.append(
            " ".join(
                words[
                    :DENSE_MAX_WORDS
                ]
            )
        )

    return np.asarray(
        output,
        dtype=object,
    )


def build_dense_index(items):
    """
    Dense embeddings объектов.

    Кэш версионируется через имя файла, чтобы новое структурированное
    представление случайно не использовало старую матрицу embeddings.
    """
    import torch

    torch.set_num_threads(
        os.cpu_count() or 8
    )

    if os.path.exists(
        DENSE_CACHE_PATH
    ):
        arr = np.load(
            DENSE_CACHE_PATH
        )

        if (
            arr.ndim == 2
            and arr.shape[0] == len(items)
            and arr.shape[1] == DENSE_DIM
        ):
            print(
                "      dense: loaded "
                f"{arr.shape}"
            )

            return arr.astype(
                "float32"
            )

    print(
        "      dense: encoding "
        f"{len(items)} items..."
    )

    model = get_dense_model()

    model.tokenizer.model_max_length = (
        DENSE_MAX_TOKENS
    )

    model.max_seq_length = (
        DENSE_MAX_TOKENS
    )

    model.eval()

    texts = [
        "passage: " + text
        for text
        in _dense_text_for_items(
            items
        ).tolist()
    ]

    embs = model.encode(
        texts,
        batch_size=DENSE_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embs = np.asarray(
        embs,
        dtype="float32",
    )

    save_arr = (
        embs.astype("float16")
        if DENSE_SAVE_FP16
        else embs
    )

    np.save(
        DENSE_CACHE_PATH,
        save_arr,
    )

    print(
        "      dense: saved "
        f"{save_arr.shape}"
    )

    return embs


# --------------------------------------------------------------------------
# Загрузка данных
# --------------------------------------------------------------------------

def load_data():
    train = pd.read_parquet(
        TRAIN_PATH
    )

    queries = pd.read_parquet(
        QUERIES_PATH
    )

    items = pd.read_parquet(
        ITEMS_PATH
    )

    items = (
        items
        .drop_duplicates(
            subset="item_id"
        )
        .reset_index(
            drop=True
        )
    )

    title = (
        items["item_title_raw"]
        .fillna("")
        .astype(str)
    )

    desc = (
        items["item_description_raw"]
        .fillna("")
        .astype(str)
    )

    params = (
        items["item_infm_params_text"]
        .fillna("")
        .astype(str)
    )

    items["item_text"] = (
        (title + " ") * TITLE_BOOST
        + desc
        + " "
        + params
    ).str.strip()

    items["item_rating"] = (
        pd.to_numeric(
            items["item_rating"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    return (
        train,
        queries,
        items,
    )


# --------------------------------------------------------------------------
# Индекс правил
# --------------------------------------------------------------------------

def build_rule_indices(
    train,
):
    """
    rule_fine:
        (q_norm, location)
            -> set(item_id)

    rule_coarse:
        (q_norm,)
            -> set(item_id)
    """
    rule_fine = {}
    rule_coarse = {}

    qn = (
        train["search_query"]
        .map(norm_query)
        .values
    )

    loc = (
        train["search_location_id"]
        .astype(int)
        .values
    )

    iid = train["item_id"].values

    for i in range(
        len(train)
    ):
        query_key = qn[i]
        location = int(loc[i])
        item_id = iid[i]

        rule_fine.setdefault(
            (
                query_key,
                location,
            ),
            set(),
        ).add(
            item_id
        )

        rule_coarse.setdefault(
            query_key,
            set(),
        ).add(
            item_id
        )

    return (
        rule_fine,
        rule_coarse,
    )


# --------------------------------------------------------------------------
# Индексы BM25
# --------------------------------------------------------------------------

def build_bm25(
    items,
):
    """
    Полнотекстовый BM25:
        title x TITLE_BOOST
        + description
        + params
    """
    import bm25s

    corpus = [
        tokenize(text)
        for text
        in items["item_text"].tolist()
    ]

    retriever = bm25s.BM25()

    retriever.index(
        corpus
    )

    return retriever


def build_bm25_title(
    items,
):
    """
    BM25 только по названию.

    Это даёт совпадению названия товара/услуги отдельный канал,
    независимый от широкого полнотекстового индекса.
    """
    import bm25s

    corpus = [
        tokenize(text)
        for text
        in (
            items["item_title_raw"]
            .fillna("")
            .astype(str)
            .tolist()
        )
    ]

    retriever = bm25s.BM25()

    retriever.index(
        corpus
    )

    return retriever


# --------------------------------------------------------------------------
# Индекс Query2Query
# --------------------------------------------------------------------------

def build_query_index(
    train,
):
    """
    BM25 по уникальным нормализованным запросам из train.
    """
    t = train[
        [
            "search_query",
            "item_id",
        ]
    ].copy()

    t["q_norm"] = (
        t["search_query"]
        .map(norm_query)
    )

    q2items = (
        t.groupby(
            "q_norm"
        )["item_id"]
        .apply(set)
        .to_dict()
    )

    unique_queries = list(
        q2items.keys()
    )

    import bm25s

    q_retriever = bm25s.BM25()

    q_retriever.index(
        [
            tokenize(q)
            for q
            in unique_queries
        ]
    )

    return (
        q_retriever,
        unique_queries,
        q2items,
    )


# --------------------------------------------------------------------------
# Индексы услуг и microcat_id
# --------------------------------------------------------------------------

def build_service_to_microcat(
    items,
):
    """
    значение услуги
        -> [microcat_id, ...]

    microcat_id
        -> [item_idx, ...]
    """
    svc_counter = defaultdict(
        Counter
    )

    mc_to_items = defaultdict(
        list
    )

    mc_arr = (
        items["item_microcat_id"]
        .values
    )

    text_arr = (
        items[
            "item_infm_params_text"
        ]
        .fillna("")
        .astype(str)
        .values
    )

    for i, (
        mc,
        text,
    ) in enumerate(
        zip(
            mc_arr,
            text_arr,
        )
    ):
        if pd.isna(mc):
            continue

        mc = int(mc)

        mc_to_items[
            mc
        ].append(i)

        for value in extract_service_values(
            text
        ):
            svc_counter[
                value
            ][mc] += 1

    svc_to_mc = {
        value: [
            mc
            for mc, _count
            in counter.most_common()
        ]
        for value, counter
        in svc_counter.items()
    }

    return (
        svc_to_mc,
        dict(mc_to_items),
    )


def build_service_value_items(
    items,
):
    """
    точное значение услуги
        -> set(item_idx)
    """
    output = defaultdict(
        set
    )

    texts = (
        items[
            "item_infm_params_text"
        ]
        .fillna("")
        .astype(str)
        .values
    )

    for i, text in enumerate(
        texts
    ):
        for value in extract_service_values(
            text
        ):
            output[value].add(i)

    return dict(output)


# --------------------------------------------------------------------------
# Построение контекста
# --------------------------------------------------------------------------

def build_context(
    train: pd.DataFrame,
    items: pd.DataFrame,
    use_cache: bool = True,
) -> PipelineContext:
    """
    Однократно построить все индексы.

    Ключи кэшей версионируются, потому что на финальном этапе разработки
    изменилось представление токенов и поиска.
    """
    if use_cache:
        fp_items = items_fingerprint(
            items
        )

        fp_train = train_fingerprint(
            train
        )

        rule_fine, rule_coarse = (
            load_or_compute(
                "rule_indices_v3",
                fp_train,
                lambda: build_rule_indices(
                    train
                ),
            )
        )

        bm25_retriever = (
            load_or_compute(
                "bm25_v3",
                fp_items,
                lambda: build_bm25(
                    items
                ),
            )
        )

        bm25_title_retriever = (
            load_or_compute(
                "bm25_title_v2",
                fp_items,
                lambda: build_bm25_title(
                    items
                ),
            )
        )

        (
            q_retriever,
            unique_queries,
            q2items,
        ) = load_or_compute(
            "query_index_v3",
            fp_train,
            lambda: build_query_index(
                train
            ),
        )

        (
            svc_to_microcat,
            microcat_items,
        ) = load_or_compute(
            "svc_to_microcat_v3",
            fp_items,
            lambda: build_service_to_microcat(
                items
            ),
        )

        service_value_items = (
            load_or_compute(
                "service_value_items_v1",
                fp_items,
                lambda: build_service_value_items(
                    items
                ),
            )
        )

    else:
        (
            rule_fine,
            rule_coarse,
        ) = build_rule_indices(
            train
        )

        bm25_retriever = build_bm25(
            items
        )

        bm25_title_retriever = (
            build_bm25_title(
                items
            )
        )

        (
            q_retriever,
            unique_queries,
            q2items,
        ) = build_query_index(
            train
        )

        (
            svc_to_microcat,
            microcat_items,
        ) = build_service_to_microcat(
            items
        )

        service_value_items = (
            build_service_value_items(
                items
            )
        )

    # Dense.
    item_embs = build_dense_index(
        items
    )

    # Lightweight arrays.
    item_id_arr = (
        items["item_id"].values
    )

    item_id_to_idx = {
        iid: i
        for i, iid
        in enumerate(
            item_id_arr
        )
    }

    item_rating = (
        items["item_rating"]
        .values
        .astype(float)
    )

    item_microcat = (
        pd.to_numeric(
            items["item_microcat_id"],
            errors="coerce",
        )
        .fillna(-1)
        .astype("int64")
        .values
    )

    item_location = (
        items["item_location_id"]
        .astype(int)
        .values
    )

    return PipelineContext(
        items_df=items,
        item_id_arr=item_id_arr,
        item_id_to_idx=item_id_to_idx,
        item_rating=item_rating,
        item_microcat=item_microcat,
        item_location=item_location,
        n_items=len(items),

        rule_fine=rule_fine,
        rule_coarse=rule_coarse,

        bm25_retriever=bm25_retriever,
        bm25_title_retriever=bm25_title_retriever,

        q_retriever=q_retriever,
        unique_queries=unique_queries,
        q2items=q2items,

        svc_to_microcat=svc_to_microcat,
        microcat_items=microcat_items,
        service_value_items=service_value_items,

        item_embs=item_embs,

        top_k=TOP_K,
        bm25_fetch_k=BM25_FETCH_K,
        rrf_k=RRF_GENERAL_K,

        min_microcat_pool=MIN_MICROCAT_POOL,
        max_microcat_pool=MAX_MICROCAT_POOL,
        microcat_bm25_max_pool=(
            MICROCAT_BM25_MAX_POOL
        ),
        dense_fetch_k=DENSE_FETCH_K,
    )


# --------------------------------------------------------------------------
# Общий поиск BM25
# --------------------------------------------------------------------------

def bm25_search(
    q_tokens,
    ctx: PipelineContext,
    k: int,
):
    """
    BM25 по всему корпусу.
    """
    if not q_tokens:
        return []

    k = min(
        int(k),
        ctx.n_items,
    )

    if k <= 0:
        return []

    results, scores = (
        ctx.bm25_retriever.retrieve(
            [q_tokens],
            k=k,
            show_progress=False,
        )
    )

    output = []

    for i, score in zip(
        results[0],
        scores[0],
    ):
        if float(score) > 0:
            output.append(
                int(i)
            )

    return output


def bm25_title_search(
    q_tokens,
    ctx: PipelineContext,
    k: int,
):
    """
    BM25 по индексу только названий.
    """
    if not q_tokens:
        return []

    k = min(
        int(k),
        ctx.n_items,
    )

    if k <= 0:
        return []

    results, scores = (
        ctx.bm25_title_retriever.retrieve(
            [q_tokens],
            k=k,
            show_progress=False,
        )
    )

    output = []

    for i, score in zip(
        results[0],
        scores[0],
    ):
        if float(score) > 0:
            output.append(
                int(i)
            )

    return output


# --------------------------------------------------------------------------
# BM25 внутри произвольного пула
# --------------------------------------------------------------------------

def bm25_search_in_pool(
    q_tokens,
    ctx: PipelineContext,
    pool_arr: np.ndarray,
    k: int,
):
    """
    Ранжировать строго внутри pool_arr по полным BM25 scores.

    Это точнее, чем:
        global top-N -> фильтрация по location

    потому что локальный объект может оказаться ниже глобального cutoff,
    но при этом быть сильным результатом внутри своего города.
    """
    if (
        not q_tokens
        or len(pool_arr) == 0
    ):
        return []

    all_scores = np.asarray(
        ctx.bm25_retriever.get_scores(
            q_tokens
        )
    )

    if all_scores.shape[0] != ctx.n_items:
        results, scores = (
            ctx.bm25_retriever.retrieve(
                [q_tokens],
                k=min(
                    len(pool_arr) + 2000,
                    ctx.n_items,
                ),
                show_progress=False,
            )
        )

        pool_set = set(
            pool_arr.tolist()
        )

        return [
            int(i)
            for i, score in zip(
                results[0],
                scores[0],
            )
            if (
                int(i) in pool_set
                and float(score) > 0
            )
        ]

    pool_scores = (
        all_scores[pool_arr]
    )

    kk = min(
        int(k),
        len(pool_arr),
    )

    if kk <= 0:
        return []

    if kk == len(pool_arr):
        order = np.argsort(
            -pool_scores
        )
    else:
        part = np.argpartition(
            -pool_scores,
            kk - 1,
        )[:kk]

        order = part[
            np.argsort(
                -pool_scores[
                    part
                ]
            )
        ]

    return pool_arr[
        order
    ].tolist()


def bm25_title_search_in_pool(
    q_tokens,
    ctx: PipelineContext,
    pool_arr: np.ndarray,
    k: int,
):
    """
    Ранжировать строго внутри pool_arr по BM25 только по названию.
    """
    if (
        not q_tokens
        or len(pool_arr) == 0
    ):
        return []

    all_scores = np.asarray(
        ctx.bm25_title_retriever.get_scores(
            q_tokens
        )
    )

    if all_scores.shape[0] != ctx.n_items:
        results, scores = (
            ctx.bm25_title_retriever.retrieve(
                [q_tokens],
                k=min(
                    len(pool_arr) + 2000,
                    ctx.n_items,
                ),
                show_progress=False,
            )
        )

        pool_set = set(
            pool_arr.tolist()
        )

        return [
            int(i)
            for i, score in zip(
                results[0],
                scores[0],
            )
            if (
                int(i) in pool_set
                and float(score) > 0
            )
        ]

    pool_scores = (
        all_scores[pool_arr]
    )

    kk = min(
        int(k),
        len(pool_arr),
    )

    if kk <= 0:
        return []

    if kk == len(pool_arr):
        order = np.argsort(
            -pool_scores
        )
    else:
        part = np.argpartition(
            -pool_scores,
            kk - 1,
        )[:kk]

        order = part[
            np.argsort(
                -pool_scores[
                    part
                ]
            )
        ]

    return pool_arr[
        order
    ].tolist()


# --------------------------------------------------------------------------
# Query2Query
# --------------------------------------------------------------------------

def query2query_candidates(
    q_tokens,
    ctx: PipelineContext,
    valid_idx_set,
    top_n_queries=20,
):
    """
    Получить объекты из похожих train-запросов.

    Кандидаты ранжируются по накопленному сходству, а не по рейтингу.
    Это сохраняет фактическую релевантность между запросами.
    """
    if not q_tokens:
        return []

    top_n_queries = min(
        int(top_n_queries),
        len(ctx.unique_queries),
    )

    if top_n_queries <= 0:
        return []

    results, scores = (
        ctx.q_retriever.retrieve(
            [q_tokens],
            k=top_n_queries,
            show_progress=False,
        )
    )

    positive_scores = [
        float(score)
        for score in scores[0]
        if float(score) > 0
    ]

    max_score = max(
        positive_scores,
        default=1.0,
    )

    item_scores = {}

    for rank, (
        qidx,
        raw_score,
    ) in enumerate(
        zip(
            results[0],
            scores[0],
        ),
        start=1,
    ):
        score = float(
            raw_score
        )

        if score <= 0:
            continue

        contribution = (
            score / max_score
        ) / (
            20.0 + rank
        )

        similar_query = (
            ctx.unique_queries[
                int(qidx)
            ]
        )

        for iid in ctx.q2items[
            similar_query
        ]:
            i = ctx.item_id_to_idx.get(
                iid
            )

            if (
                i is None
                or i not in valid_idx_set
            ):
                continue

            item_scores[i] = (
                item_scores.get(
                    i,
                    0.0,
                )
                + contribution
            )

    ordered = sorted(
        item_scores.items(),
        key=lambda pair: (
            -pair[1],
            ctx.item_id_arr[
                pair[0]
            ],
        ),
    )

    return [
        i
        for i, _score
        in ordered
    ]


# --------------------------------------------------------------------------
# Представление запроса для dense-поиска
# --------------------------------------------------------------------------

def query_text_for_dense(
    q,
) -> str:
    """
    Структурированное представление запроса для dense:
        search_query
        + точные значения услуги

    Общие названия параметров не используются.
    """
    parts = []

    query_text = str(
        _safe_attr(
            q,
            "search_query",
            "",
        )
        or ""
    ).strip()

    if query_text:
        parts.append(
            query_text
        )

    infm = _safe_attr(
        q,
        "search_infm_params_text",
        None,
    )

    service_values = (
        extract_service_values(
            infm
        )
    )

    parts.extend(
        service_values
    )

    return " ".join(
        x
        for x in parts
        if x
    ).strip()


def dense_search(
    q,
    ctx: PipelineContext,
    k: int,
):
    """
    Косинусное сходство dense-векторов по всему корпусу.
    """
    if (
        ctx.item_embs is None
        or len(ctx.item_embs) == 0
    ):
        return []

    q_text = query_text_for_dense(
        q
    )

    if not q_text:
        return []

    model = get_dense_model()

    q_emb = model.encode(
        [
            "query: "
            + q_text
        ],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    q_emb = np.asarray(
        q_emb,
        dtype="float32",
    )[0]

    sims = (
        ctx.item_embs
        @ q_emb
    )

    k = min(
        int(k),
        len(sims),
    )

    if k <= 0:
        return []

    if k == len(sims):
        order = np.argsort(
            -sims
        )
    else:
        part = np.argpartition(
            -sims,
            k - 1,
        )[:k]

        order = part[
            np.argsort(
                -sims[
                    part
                ]
            )
        ]

    return [
        int(i)
        for i in order
    ]


# --------------------------------------------------------------------------
# Кандидаты по точному значению услуги
# --------------------------------------------------------------------------

def service_exact_candidates(
    q,
    ctx: PipelineContext,
    q_tokens_bm25,
    k,
):
    """
    Источник кандидатов по точному значению услуги.

    Кандидатный набор не фильтруется жёстко. Сначала находятся объекты с тем же
    нормализованным значением услуги, затем они ранжируются полнотекстовым BM25.
    """
    infm = _safe_attr(
        q,
        "search_infm_params_text",
        None,
    )

    values = extract_service_values(
        infm
    )

    if not values:
        return []

    pool = set()

    for value in values:
        pool.update(
            ctx.service_value_items.get(
                value,
                (),
            )
        )

    if not pool:
        return []

    # Avoid overly broad exact-value pools.
    if len(pool) > 30000:
        return []

    pool_arr = np.fromiter(
        pool,
        dtype=np.int64,
        count=len(pool),
    )

    return bm25_search_in_pool(
        q_tokens_bm25,
        ctx,
        pool_arr,
        k,
    )


# --------------------------------------------------------------------------
# Пул microcat_id
# --------------------------------------------------------------------------

def compute_microcat_pool(
    q,
    ctx,
    allowed_idx_set,
    min_pool=MIN_MICROCAT_POOL,
    max_pool=MAX_MICROCAT_POOL,
):
    """
    Построить microcategory-кандидатов из:

      A. значений услуги, явно присутствующих в параметрах запроса;
      B. microcategory объектов train, наблюдавшихся для того же запроса.

    Если доступны оба источника:
      сначала используется пересечение;
      если оно пусто — используется объединение.
    """
    query_norm = norm_query(
        _safe_attr(
            q,
            "search_query",
            "",
        )
        or ""
    )

    mcs_from_query = set()

    infm = _safe_attr(
        q,
        "search_infm_params_text",
        None,
    )

    for value in extract_service_values(
        infm
    ):
        mcs_from_query.update(
            ctx.svc_to_microcat.get(
                value,
                (),
            )
        )

    mcs_from_train = set()

    for iid in ctx.rule_coarse.get(
        query_norm,
        (),
    ):
        i = ctx.item_id_to_idx.get(
            iid
        )

        if i is None:
            continue

        mc = int(
            ctx.item_microcat[i]
        )

        if mc >= 0:
            mcs_from_train.add(
                mc
            )

    if (
        mcs_from_query
        and mcs_from_train
    ):
        target = (
            mcs_from_query
            & mcs_from_train
        )

        if not target:
            target = (
                mcs_from_query
                | mcs_from_train
            )

    elif mcs_from_query:
        target = mcs_from_query

    elif mcs_from_train:
        target = mcs_from_train

    else:
        return None

    pool = set()

    for mc in target:
        for i in ctx.microcat_items.get(
            mc,
            (),
        ):
            if i in allowed_idx_set:
                pool.add(i)

    if (
        len(pool) < min_pool
        or len(pool) > max_pool
    ):
        return None

    return pool


# --------------------------------------------------------------------------
# Query tokens
# --------------------------------------------------------------------------

_INFM_KEY_STOPWORDS = {
    "вид",
    "услуга",
    "тип",
    "оказание",
    "онлайн",
    "запись",
}


def query_tokens_for_bm25(
    q,
):
    """
    Полный BM25-запрос:
        search_query
        + полезные значения параметров

    Общие структурные названия полей удаляются.
    """
    q_str = str(
        _safe_attr(
            q,
            "search_query",
            "",
        )
        or ""
    )

    tokens = tokenize(
        q_str
    )

    infm = _safe_attr(
        q,
        "search_infm_params_text",
        None,
    )

    if (
        isinstance(
            infm,
            str,
        )
        and infm.strip()
    ):
        infm_tokens = [
            token
            for token
            in tokenize(infm)
            if token
            not in _INFM_KEY_STOPWORDS
        ]

        tokens.extend(
            infm_tokens
        )

    return tokens


# --------------------------------------------------------------------------
# Маршрутизация RRF
# --------------------------------------------------------------------------

def get_rrf_config(
    q,
    ctx: PipelineContext,
):
    """
    Двухрежимная маршрутизация:

      seen_location:
          q_norm встречался в train И точный location встречался в train

      general:
          всё остальное
    """
    query_text = str(
        _safe_attr(
            q,
            "search_query",
            "",
        )
        or ""
    )

    qn = norm_query(
        query_text
    )

    q_loc = int(
        _safe_attr(
            q,
            "search_location_id",
            -1,
        )
    )

    query_seen = bool(
        ctx.rule_coarse.get(
            qn,
            set(),
        )
    )

    query_location_seen = bool(
        ctx.rule_fine.get(
            (
                qn,
                q_loc,
            ),
            set(),
        )
    )

    if (
        query_seen
        and query_location_seen
    ):
        return {
            "mode": "seen_location",
            "rrf_k": RRF_SEEN_K,

            "rule_fine":
                W_SEEN_RULE_FINE,
            "rule_coarse":
                W_SEEN_RULE_COARSE,

            "bm25_local":
                W_SEEN_BM25_LOCAL,
            "bm25_global":
                W_SEEN_BM25_GLOBAL,

            "title_local":
                W_SEEN_TITLE_LOCAL,
            "title_global":
                W_SEEN_TITLE_GLOBAL,

            "service_exact":
                W_SEEN_SERVICE_EXACT,

            "query2query":
                W_SEEN_QUERY2QUERY,
            "dense":
                W_SEEN_DENSE,

            "microcat":
                W_SEEN_MICROCAT,

            "query_seen": True,
            "query_location_seen": True,
        }

    return {
        "mode": "general",
        "rrf_k": RRF_GENERAL_K,

        "rule_fine":
            W_GENERAL_RULE_FINE,
        "rule_coarse":
            W_GENERAL_RULE_COARSE,

        "bm25_local":
            W_GENERAL_BM25_LOCAL,
        "bm25_global":
            W_GENERAL_BM25_GLOBAL,

        "title_local":
            W_GENERAL_TITLE_LOCAL,
        "title_global":
            W_GENERAL_TITLE_GLOBAL,

        "service_exact":
            W_GENERAL_SERVICE_EXACT,

        "query2query":
            W_GENERAL_QUERY2QUERY,
        "dense":
            W_GENERAL_DENSE,

        "microcat":
            W_GENERAL_MICROCAT,

        "query_seen": query_seen,
        "query_location_seen":
            query_location_seen,
    }


# --------------------------------------------------------------------------
# Основной генератор кандидатов
# --------------------------------------------------------------------------

def generate_candidates_for_query(
    q,
    ctx: PipelineContext,
    top_k: int | None = None,
    return_sources: bool = False,
):
    """
    Сгенерировать до top_k финальных кандидатов.
    """
    top_k = (
        top_k
        if top_k is not None
        else ctx.top_k
    )

    q_text = str(
        _safe_attr(
            q,
            "search_query",
            "",
        )
        or ""
    )

    qn = norm_query(
        q_text
    )

    q_loc = int(
        _safe_attr(
            q,
            "search_location_id",
            -1,
        )
    )

    all_idx_set = set(
        range(ctx.n_items)
    )

    # --------------------------------------------------------------
    # Маршрутизация
    # --------------------------------------------------------------

    cfg = get_rrf_config(
        q,
        ctx,
    )

    # --------------------------------------------------------------
    # Пул локации
    # --------------------------------------------------------------

    loc_valid = np.where(
        ctx.item_location == q_loc
    )[0]

    if len(loc_valid) >= top_k:
        loc_pool = np.asarray(
            loc_valid,
            dtype=np.int64,
        )

        loc_valid_set = set(
            loc_valid.tolist()
        )
    else:
        # Для маленьких городов используется весь корпус
        loc_pool = np.arange(
            ctx.n_items,
            dtype=np.int64,
        )

        loc_valid_set = all_idx_set

    # --------------------------------------------------------------
    # Токены запроса
    # --------------------------------------------------------------

    q_tokens_bm25 = (
        query_tokens_for_bm25(q)
    )

    q_tokens_query = tokenize(
        q_text
    )

    # --------------------------------------------------------------
    # Правила
    # --------------------------------------------------------------

    rf_ids = ctx.rule_fine.get(
        (
            qn,
            q_loc,
        ),
        set(),
    )

    rf_idx = [
        ctx.item_id_to_idx[iid]
        for iid in rf_ids
        if iid in ctx.item_id_to_idx
    ]

    # Детерминированный подход
    rf_idx.sort(
        key=lambda i:
        ctx.item_id_arr[i]
    )

    rc_ids = ctx.rule_coarse.get(
        qn,
        set(),
    )

    rc_idx = [
        ctx.item_id_to_idx[iid]
        for iid in rc_ids
        if iid in ctx.item_id_to_idx
    ]

    rc_idx.sort(
        key=lambda i:
        ctx.item_id_arr[i]
    )

    # --------------------------------------------------------------
    # Полнотекстовый BM25
    # --------------------------------------------------------------

    # Глобальный поиск всегда выполняется как независимый
    # канал восстановления.
    bm25_global = bm25_search(
        q_tokens_bm25,
        ctx,
        ctx.bm25_fetch_k,
    )

    # Локальный BM25 является основным лексическим источником в обоих режимах маршрутизации.
    bm25_local = bm25_search_in_pool(
        q_tokens_bm25,
        ctx,
        loc_pool,
        ctx.bm25_fetch_k,
    )

    # --------------------------------------------------------------
    # BM25 по названию
    # --------------------------------------------------------------

    title_global = bm25_title_search(
        q_tokens_query,
        ctx,
        ctx.bm25_fetch_k,
    )

    title_local = bm25_title_search_in_pool(
        q_tokens_query,
        ctx,
        loc_pool,
        ctx.bm25_fetch_k,
    )

    # --------------------------------------------------------------
    # Поиск по точному значению услуги
    # --------------------------------------------------------------

    service_exact = (
        service_exact_candidates(
            q,
            ctx,
            q_tokens_bm25,
            ctx.bm25_fetch_k,
        )
    )

    # --------------------------------------------------------------
    # microcat_id
    # --------------------------------------------------------------

    microcat_pool = (
        compute_microcat_pool(
            q,
            ctx,
            all_idx_set,
        )
    )

    bm25_microcat = []

    if (
        microcat_pool is not None
        and len(microcat_pool)
        <= ctx.microcat_bm25_max_pool
    ):
        pool_arr = np.fromiter(
            microcat_pool,
            dtype=np.int64,
            count=len(microcat_pool),
        )

        bm25_microcat = (
            bm25_search_in_pool(
                q_tokens_bm25,
                ctx,
                pool_arr,
                ctx.bm25_fetch_k,
            )
        )

    # --------------------------------------------------------------
    # Query2Query
    # --------------------------------------------------------------

    q2q_idx = (
        query2query_candidates(
            q_tokens_query,
            ctx,
            all_idx_set,
            top_n_queries=20,
        )
    )

    # --------------------------------------------------------------
    # Векторные представления (Dense)
    # --------------------------------------------------------------

    dense_idx = dense_search(
        q,
        ctx,
        ctx.dense_fetch_k,
    )

    # --------------------------------------------------------------
    # Объединение RRF (Fusion)
    # --------------------------------------------------------------

    sources = (
        (
            rf_idx[
                :MAX_SOURCE_LEN
            ],
            cfg["rule_fine"],
        ),
        (
            rc_idx[
                :MAX_SOURCE_LEN
            ],
            cfg["rule_coarse"],
        ),
        (
            bm25_local[
                :MAX_SOURCE_LEN
            ],
            cfg["bm25_local"],
        ),
        (
            bm25_global[
                :MAX_SOURCE_LEN
            ],
            cfg["bm25_global"],
        ),
        (
            title_local[
                :MAX_SOURCE_LEN
            ],
            cfg["title_local"],
        ),
        (
            title_global[
                :MAX_SOURCE_LEN
            ],
            cfg["title_global"],
        ),
        (
            service_exact[
                :MAX_SOURCE_LEN
            ],
            cfg["service_exact"],
        ),
        (
            bm25_microcat[
                :MAX_SOURCE_LEN
            ],
            cfg["microcat"],
        ),
        (
            q2q_idx[
                :MAX_SOURCE_LEN
            ],
            cfg["query2query"],
        ),
        (
            dense_idx[
                :MAX_SOURCE_LEN
            ],
            cfg["dense"],
        ),
    )

    rrf = {}

    for ranking, weight in sources:
        if weight <= 0:
            continue

        for rank, idx in enumerate(
            ranking,
            start=1,
        ):
            iid = ctx.item_id_arr[
                idx
            ]

            rrf[iid] = (
                rrf.get(
                    iid,
                    0.0,
                )
                + (
                    weight
                    / (
                        cfg["rrf_k"]
                        + rank
                    )
                )
            )

    # --------------------------------------------------------------
    # Финальное детерминированное ранжирование
    # --------------------------------------------------------------

    fused = sorted(
        rrf.items(),
        key=lambda pair: (
            -pair[1],
            pair[0],
        ),
    )

    top_items = [
        iid
        for iid, _score
        in fused[
            :top_k
        ]
    ]

    # --------------------------------------------------------------
    # Дополнение списка
    # --------------------------------------------------------------

    if len(top_items) < top_k:
        seen = set(
            top_items
        )

        # Сначала использовать остатки глобального BM25, поскольку это самый широкий лексический
        # источник и потому он безопаснее произвольного дополнения по item_id
        for idx in bm25_global:
            iid = ctx.item_id_arr[
                idx
            ]

            if iid in seen:
                continue

            seen.add(
                iid
            )

            top_items.append(
                iid
            )

            if len(top_items) >= top_k:
                break

    if len(top_items) < top_k:
        # Затем локальный пул
        for idx in bm25_local:
            iid = ctx.item_id_arr[
                idx
            ]

            if iid in seen:
                continue

            seen.add(
                iid
            )

            top_items.append(
                iid
            )

            if len(top_items) >= top_k:
                break

    if len(top_items) < top_k:
        # Детерминированное дополнение из корпуса
        for idx in range(
            ctx.n_items
        ):
            iid = ctx.item_id_arr[
                idx
            ]

            if iid in seen:
                continue

            seen.add(
                iid
            )

            top_items.append(
                iid
            )

            if len(top_items) >= top_k:
                break

    # --------------------------------------------------------------
    # Возвращение результата
    # --------------------------------------------------------------

    if not return_sources:
        return top_items

    def to_ids(
        indices,
    ):
        return [
            ctx.item_id_arr[i]
            for i in indices
        ]

    return {
        "top_items": top_items,

        "sources": {
            "rule_fine":
                to_ids(rf_idx),

            "rule_coarse":
                to_ids(rc_idx),

            "bm25":
                to_ids(bm25_local),

            "bm25_local":
                to_ids(bm25_local),

            "bm25_global":
                to_ids(bm25_global),

            "title_local":
                to_ids(title_local),

            "title_global":
                to_ids(title_global),

            "service_exact":
                to_ids(service_exact),

            "bm25_microcat":
                to_ids(bm25_microcat),

            "query2query":
                to_ids(q2q_idx),

            "dense":
                to_ids(dense_idx),

            "microcat": (
                to_ids(
                    sorted(
                        microcat_pool
                    )
                )
                if microcat_pool
                else []
            ),
        },

        "valid_count":
            len(loc_valid_set),

        "microcat_active":
            microcat_pool is not None,

        "rrf_mode":
            cfg["mode"],

        "query_seen":
            cfg["query_seen"],

        "query_location_seen":
            cfg[
                "query_location_seen"
            ],

        "rrf_config":
            cfg,
    }
