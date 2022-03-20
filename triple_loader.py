

def read_triples(file_name: str) -> list[tuple[int, int, int]]:
    triple_store = []
    with open(file_name, 'rb') as f:
        for line in f:
            (s, p, o, dot) = line.split(None)
            triple_store.append((s, p, o))

    return triple_store
