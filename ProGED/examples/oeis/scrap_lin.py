"""
File that scrapes webpage:
    http://oeis.org/wiki/Index_to_OEIS:_Section_Rec
for linearly recursive sequences.
Good for discovering exact linear integer equations by algorithm that solves
diophantine equations.

list of ids.
for id in list:
    seq = blist[id]

"""

import requests, re
import lxml
from bs4 import BeautifulSoup

# url = "https://www.rottentomatoes.com/top/bestofrt/"
url = "http://oeis.org/wiki/Index_to_OEIS:_Section_Rec"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, '
                  'like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'
}
f = requests.get(url, headers = headers)
# print(f.content)
# 1/0
movies_lst = []
# soup = BeautifulSoup(f.content, 'lxml')
soup = BeautifulSoup(f.content, 'html.parser')
print(soup.prettify())
print(soup)

subsects = soup.find_all(class_='mw-headline')
# print(subsect)

# seqs = dict()
orders = []
sects_dict = dict()
c = -1
for subsect in subsects:
    title = subsect.text
    # order = re.findall(r'(\d+)', title)[0] if len(re.findall(r'(\d+)', title))> 0 else None
    order_set = re.findall(r'(\d+)', title)
    if len(order_set) == 0:
        pass
        # sects_dict[str(c)] = subsect
        # c -= 1
    else:
        sects_dict[order_set[0]] = subsect
    # sects_dict[str(order)] = subsect
    # orders += [order]
    # seqs[order]
    # print(order)
# print(len(orders))

degen = [(k, v) for k, v in sects_dict.items() if int(k) < 0]
print(len(degen))
for k, v in degen:
    print(k, v)

for k, v in sects_dict.items():
    print(k, v)


# idea:
# if previous_sibling == h4:
#     separate here



ids = soup.find_all('a', text=re.compile(r'A\w{6}'))
# ids = soup.find_all(text=r'A\w{6}')
# ids_txt = [i.text for i in ids]
ids_txt = ids
print(len(ids_txt))
print(len(ids))
print(len(set(ids_txt)))
found = re.findall(r'A\w{6}', 'dsadsa dsadsa2323sats A232423 dsa sd0a dsa')
print(found)




counter = dict()
for id_ in ids_txt:
    # if ids_txt[id_] += 1
    # counter[id_] = 1 \
    if id_ not in counter:
        counter[id_] = 1
    else:
        counter[id_] += 1

doublets =  [(k, i) for k, i in counter.items() if i>1]
# print(len())



# id_.previous

linseqs = dict()

# id_ = ids[1]
print(len(ids))
for id_ in ids[:10204]:
    parent = id_.parent
    if parent.previous_sibling is None:
        previous = parent.parent.previous_sibling.previous_sibling
        if previous.name == 'h4':
        # title = id_.parent.parent.previous_sibling.previous_sibling.text
            title = previous.text
            order = re.findall(r'(\d+)', title)[0]
    # if order not in linseqs:
            linseqs[order] = [id_.text]
        elif previous.name not in linseqs:
            linseqs[previous.name] = [previous]
        else:
            linseqs[previous.name] += [previous]
    else:
        linseqs[order] += [id_.text]

print(len(linseqs))
print([(seqs, len(linseqs[seqs])) for seqs in linseqs])
print(sum(len(linseqs[seqs]) for seqs in linseqs))

for order, seqs in linseqs.items():
    print(f'order: {order}')
    for seq in seqs:
        # print(int(order) * "  " + f'\\_ {seq}   order: {order}')
        print("  " + f'\\_ {seq}   order: {order}')

# before.parent.previous_sibling.previous_sibling







# soup.find_all('link')
tag = soup.find_all('p')
prices = soup.find_all(text='$')
parent = prices[0].parent
name = parent.name
strong = parent.find('strong')

specific_parent = prices[0].find_parent(class_='item-container')

tag['color'] = "blue"
print(tag.attrs)
print(tag.attrs)
for tag in tags:
    print(tag.strip())  # '\n    $1233    ' -> '$123'

# Search in all tags inside of the list:
tags = soup.find_all(['option', 'div', 'li'], text='Undergraduate', value="undergraduate")
tags = soup.find_all(class_='btn-value')
tags = soup.find_all(text=re.compile("\$.*"))

tbody = soup.tbody  # <tbody>
trs = tbody.contents
trs[0].next_sibling
trs[0].previous_sibling
trs[0].next_siblings
trs[0].contents
trs[0].descendents



1/0
movies = soup.find('table', {
    movies = soup.find('table', {
    'class': 'table'
}).find_all('a')
num = 0
for anchor in movies:
    urls = 'https://www.rottentomatoes.com' + anchor['href']
movies_lst.append(urls)
num += 1
movie_url = urls
movie_f = requests.get(movie_url, headers = headers)
movie_soup = BeautifulSoup(movie_f.content, 'lxml')
movie_content = movie_soup.find('div', {
    'class': 'movie_synopsis clamp clamp-6 js-clamp'
})
print(num, urls, '\n', 'Movie:' + anchor.string.strip())
print('Movie info:' + movie_content.string.strip())


