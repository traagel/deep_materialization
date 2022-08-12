from shlex import split


def read_encoded_triples(file_name: str) -> set[tuple[int, int, int]]:
    encoded_triple_store = set()
    with open(file_name, 'rb') as f:
        for line in f:
            (s, p, o) = split(line.decode("utf-8"))
            encoded_triple_store.add((int(s), int(p), int(o)))

    return encoded_triple_store


def read_triples(file_name: str) -> list[tuple[str, str, str]]:
    triple_store = []
    with open(file_name, 'rb') as f:
        for line in f:
            (s, p, o, dot) = split(line.decode("utf-8"))
            triple_store.append((s, p, o))

    return triple_store
