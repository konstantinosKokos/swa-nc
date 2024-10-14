from definitions import Nomino
import json

numstrings = tuple(map(str, range(1, 20)))
consecutives = tuple(zip(numstrings, numstrings[1:]))


def find_infix(string: str | None, left: str | None, right: str | None) -> str | None:
    if string is None:
        return None
    match left, right:
        case None, None:
            return string
        case _, None:
            return string.split(left)[1] if left in string else None
        case None, _:
            return string.split(right)[0] if right in string else None
        case _, _:
            return find_infix(find_infix(string, left, None), None, right)


def extract_concord(x: str) -> str:
    return find_infix(x, left='[', right=']')


def extract_definitions(x: str) -> tuple[str, ...]:
    content = find_infix(x, left=']', right=None)
    if content is None:
        return ()
    definitions = ()
    for count, (left, right) in enumerate(consecutives):
        definition = find_infix(content, left, right)
        if definition is None:
            break
        definitions = (*definitions, clean_definition(definition))
    if not definitions:
        return (clean_definition(content), )
    definitions = (
        *definitions,
        clean_definition(find_infix(content, left=consecutives[count - 1][1], right=None)))  # noqa
    return definitions


def clean_definition(x: str) -> str | None:
    if '~' in x and ':' not in x:
        return None
    if (y := find_infix(x, left=None, right=':')) is not None:
        x = y
    if (y := find_infix(x, left=None, right='(')) is not None:
        x = y
    return x.strip().rstrip('.')


def parse_entry(entry: str, entry_string: str) -> tuple[Nomino, ...]:
    concord = extract_concord(entry_string)
    definitions = extract_definitions(entry_string)
    return tuple(
        Nomino(
            entry=entry,
            definition=definition,
            subject_concord=concord)
        for definition in definitions)


if __name__ == '__main__':
    with open('../data/crawled.json', 'r') as f:
        parsed = json.load(f)
    print(f'Read {len(parsed)} entries.')

    nominos = [nom for k, vs in parsed.items() for v in vs for nom in parse_entry(k, v)
               if nom.definition is not None and nom.definition
               and nom.subject_concord is not None and nom.subject_concord]
    print(f'Parsed into {len(nominos)} nominos.')

    with open('../data/parsed.json', 'w') as f:
        json.dump([nomino.to_json() for nomino in nominos], f, indent=4)
