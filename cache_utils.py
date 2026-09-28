import hashlib
import os
import pickle

DATA_DIR = os.environ.get("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))

_CACHE_DIR = os.path.join(DATA_DIR, ".cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _fingerprint_to_name(name: str, fp: str) -> str:
    return f"{name}_{fp[:12]}.pkl"


def load_or_compute(name: str, fingerprint: str, builder, verbose: bool = True):
    """Загрузить `name` из дискового кэша, если fingerprint совпадает; иначе пересчитать.

    Параметры
    ---------
    name : str
        Логическое имя артефакта (например, "bm25", "query_index").
    fingerprint : str
        Хэш или строка, однозначно идентифицирующая входные данные. Два запуска
        с одинаковыми входными данными используют общий кэш; любое изменение
        инвалидирует его.
    builder : callable
        Функция без аргументов, возвращающая объект для кэширования.
    """
    path = os.path.join(_CACHE_DIR, _fingerprint_to_name(name, fingerprint))
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if verbose:
                print(f"      [cache] {name}: hit ({os.path.basename(path)})")
            return obj
        except Exception as e:  # noqa: BLE001
            print(f"      [cache] {name}: load failed ({e}), recomputing")

    if verbose:
        print(f"      [cache] {name}: miss, computing...")
    obj = builder()
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    if verbose:
        print(f"      [cache] {name}: saved ({os.path.basename(path)})")
    return obj


# --------------------------------------------------------------------------
# Отпечатки общих входных данных
# --------------------------------------------------------------------------
def items_fingerprint(items) -> str:
    """Короткий хэш, идентифицирующий корпус объектов."""
    h = hashlib.md5()
    h.update(str(len(items)).encode())
    if len(items):
        h.update(str(items["item_id"].iloc[0]).encode())
        h.update(str(items["item_id"].iloc[-1]).encode())
        # включаем ratings, чтобы их изменение инвалидировало кэш
        h.update(str(float(items["item_rating"].sum())).encode())
    return h.hexdigest()


def train_fingerprint(train) -> str:
    """Короткий хэш, идентифицирующий train-выборку."""
    h = hashlib.md5()
    h.update(str(len(train)).encode())
    if len(train):
        h.update(str(train["item_id"].iloc[0]).encode())
        h.update(str(train["item_id"].iloc[-1]).encode())
        h.update(str(train["search_query"].nunique()).encode())
    return h.hexdigest()