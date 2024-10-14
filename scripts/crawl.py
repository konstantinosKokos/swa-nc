import os
import json
import requests
from bs4 import BeautifulSoup

base = 'https://swahili-dictionary.com'

start_urls = [
    f'https://swahili-dictionary.com/i/sw/swahili-nouns-english-translation-{x}.html'
    for x in 'abcdefghijklmnopqrstuvwxyz']


def get_entries():
    for url in start_urls:
        print(url)
        response = requests.get(url)
        html_content = response.content

        soup = BeautifulSoup(html_content, 'html.parser')

        try:
            list_items = soup.find('body').\
                find('main').\
                find('section', {'id': 'main-content', 'role': 'main'}).\
                find('div', {'class': 'page indices'}).\
                find('ul', {'class': 'index-list'}).\
                find_all('li')

            for item in list_items:
                link = item.find("a")
                href = link.get("href")
                yield href
        except AttributeError:
            print(' --Failed')
            yield from ()


def parse_entry_html(url: str) -> tuple[str, tuple[str, ...]]:
    response = requests.get(url)
    html_content = response.content

    soup = BeautifulSoup(html_content, 'html.parser')

    try:
        main_content = soup.find('body').\
            find('main').\
            find('div', {'class': 'main-container'}).\
            find('section', {'id': 'main-content'})

        entry_title = main_content.find('h1').get_text(strip=True)
        contents = main_content.find_all('article')

        return entry_title, tuple(content.get_text(strip=False) for content in contents)

    except AttributeError:
        return '', ()


if __name__ == '__main__':
    def to_url(x: str) -> str: return base + x

    link_path = '../data/links.txt'
    if not os.path.exists(link_path):
        print('Extracting links...')
        links = list(map(to_url, get_entries()))
        print('Storing links...')
        with open(link_path, 'w') as f:
            f.write('\n'.join(links))
    print('Reading links...')
    with open(link_path, 'r') as f:
        links = f.read().split('\n')

    crawl_path = '../data/crawled.json'
    if os.path.exists(crawl_path):
        'Reading parsed..'
        with open(crawl_path, 'r') as f:
            crawled = json.load(f)
            idx = int(crawled['idx'])
    else:
        crawled = {}
        idx = 0
    print(f'Starting from {idx} ...')
    for i, (title, content) in enumerate(map(parse_entry_html, links[idx:])):
        content = tuple(c for c in content if c)
        if title and content:
            crawled[title] = content
            crawled['idx'] = str(idx + i)
            with open(crawl_path, 'w') as f:
                json.dump(crawled, f, indent=4)
