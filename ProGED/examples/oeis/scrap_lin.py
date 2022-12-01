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
import os

import requests, re, time
import pandas as pd
import lxml
from bs4 import BeautifulSoup

# url = "https://www.rottentomatoes.com/top/bestofrt/"
url = "http://oeis.org/wiki/Index_to_OEIS:_Section_Rec"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, '
                  'like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'
}
f = requests.get(url, headers = headers)
# f = requests.get(url)
# print(f.content)
# 1/0
movies_lst = []
# soup = BeautifulSoup(f.content, 'lxml')
soup = BeautifulSoup(f.content, 'html.parser')
# print(soup.prettify())
# print(soup)


# subsects = soup.find_all(class_='mw-headline')
# # print(subsect)
#
# # seqs = dict()
# orders = []
# sects_dict = dict()
# c = -1
# for subsect in subsects:
#     title = subsect.text
#     # order = re.findall(r'(\d+)', title)[0] if len(re.findall(r'(\d+)', title))> 0 else None
#     order_set = re.findall(r'(\d+)', title)
#     if len(order_set) == 0:
#         pass
#         # sects_dict[str(c)] = subsect
#         # c -= 1
#     else:
#         sects_dict[order_set[0]] = subsect
#     # sects_dict[str(order)] = subsect
#     # orders += [order]
#     # seqs[order]
#     # print(order)
# # print(len(orders))
#
# degen = [(k, v) for k, v in sects_dict.items() if int(k) < 0]
# print(len(degen))
# for k, v in degen:
#     print(k, v)
#
# for k, v in sects_dict.items():
#     print(k, v)
#
#
# # idea:
# # if previous_sibling == h4:
# #     separate here



ids = soup.find_all('a', text=re.compile(r'A\d{6}'))
# ids = soup.find_all(text=r'A\w{6}')
# ids_txt = [i.text for i in ids]
ids_txt = ids
print(len(ids_txt))
print(len(ids))
print(len(set(ids_txt)))
# found = re.findall(r'A\w{6}', 'dsadsa dsadsa2323sats A232423 dsa sd0a dsa')
# print(found)



# counter = dict()
# for id_ in ids_txt:
#     # if ids_txt[id_] += 1
#     # counter[id_] = 1 \
#     if id_ not in counter:
#         counter[id_] = 1
#     else:
#         counter[id_] += 1
# doublets =  [(k, i) for k, i in counter.items() if i>1]
# # print(len())
# id_.previous
# id_ = ids[1]
print(len(ids))

"""
linseqs = dictionary of orders as keys and values as lists of seqs of linear recursive order.
idea: 
  linseqs -> seqsd = dict( id: seq) -> pd.Dataframe(seqsd).sorted.to_csv() 
 
for seq in seqs: bfile2list

df = pd.DataFrame(seqs)
df_sorted = df.sort_index(axis=1)
print(df_sorted.head())
# csv_filename = "oeis_selection.csv"  # Masters
csv_filename = "oeis_dmkd.csv"
# df_sorted.to_csv(csv_filename, index=False)
"""


# start = 100
start = 13800
linseqs = dict()
scale_tree = 10
# scale_tree = 1000
# scale_tree = 30000
# scale_tree = 10**5
# scale_tree = 40000
# scale_tree = 25000
scale_tree = 1000000
# print(scale_tree)
# linseqs['7'] = [None]
# scale_tree = 13900

# linseqs['131071'] = [None]
start = 0
# scale_tree = 10**6

# order = '8'
# linseqs['8'] = [None]

for id_ in ids[start:scale_tree]:
    parent = id_.parent
    if parent.previous_sibling is None:
        previous = parent.parent.previous_sibling.previous_sibling
        if previous.name == 'h4':
        # title = id_.parent.parent.previous_sibling.previous_sibling.text
            title = previous.text
            order = re.findall(r'(\d+)', title)[0]
    # if order not in linseqs:
            linseqs[order] = [id_.text]
        else:
            linseqs[order] += [id_.text]
            if previous.name not in linseqs:
                pass
                # linseqs[previous.name] = [previous]
            else:
                pass
                # linseqs[previous.name] += [previous]
    else:
        linseqs[order] += [id_.text]


print(len(linseqs))
print([(seqs, len(linseqs[seqs])) for seqs in linseqs])
print(sum(len(linseqs[seqs]) for seqs in linseqs))

ids_list = []
for _, seqs in linseqs.items():
    ids_list += seqs
print(ids_list, len(ids_list))


reconst = []
for seqs in linseqs.values():
    reconst += seqs
print(len(reconst))
ids_raw = {id_.text for id_ in ids[start:scale_tree]}
reconsts = set(reconst)
print(set(ids_raw).difference(reconsts))
# print(ids_raw)
# print(reconst[:14])



verbosity = False
# verbosity = True
if verbosity:
    for order, seqs in linseqs.items():
        print(f'order: {order}')
        for seq in seqs:
            # print(int(order) * "  " + f'\\_ {seq}   order: {order}')
            print("  " + f'\\_ {seq}   order: {order}')



# before.parent.previous_sibling.previous_sibling

# linseqs


now = time.perf_counter()
def timer(now, text=f"\nScraping all (chosen) OEIS sequences"):
    before, now = now, time.perf_counter()
    consumed = now - before
    print(text +
          f" took:\n {round(consumed, 1)} secconds,"
          f" i.e. {round(consumed / 60, 2)} minutes"
          f" or {round(consumed / 3600, 3)} hours.")
    return now


# create csv
if os.getcwd()[-11:] == 'ProGED_oeis':
    from ProGED.examples.oeis.scraping.downloading.download import bfile2list
else:
    from scraping.downloading.download import bfile2list

# sa009 = bfile2list('A009117', 200)


csv_filename = "linear_database.csv"


# seqs_dict = dict()
to_concat = []
scale_per_ord = 1000000
scale_per_ord = 100
SCALE_COUNT = 10
SCALE_COUNT = 5000
counter = 0
escape = False
PERIOD = 100
# PERIOD = 5

print(f'scale_per_ord:{scale_per_ord}')
print(f'SCALE_COUNT:{SCALE_COUNT}')
print(f'PERIOD:{PERIOD}')
# scale_per_ord = 10
MAX_SEQ_LENGTH = 200
# MAX_SEQ_LENGTH = 6
# MAX_SEQ_LENGTH = 15

PARALLELIZE = True
PARALLELIZE = False
PARALEL_BATCH = 500
INDEX = 500
if PARALLELIZE:
    sorted_ids = sorted(ids_list)
    for id in sorted_ids:
        to_concat += [pd.DataFrame({id_: [int(an) for an in bfile2list(id_, max_seq_length=MAX_SEQ_LENGTH)]})]
    counter += 1
    if counter % PERIOD == 0:
        timer(now, f"Scraping one of the parallel batches of {counter} sequences ")
        print(f"counter: {counter}")
        df = pd.concat(to_concat, axis=1)
        df.sort_index(axis=1).to_csv(csv_filename, index=False)
        print(f"{counter} sequences written to csv")
        print("check file: number of ids, file size?")
else:
    for order, ids in linseqs.items():
        if verbosity >= 2:
            print(order)
            print(ids[:scale_per_ord])
        for id_ in ids[:scale_per_ord]:
            # print([type(i) for i in [idii, idsii, orderii]])

            if counter <= SCALE_COUNT:
                to_concat += [pd.DataFrame({id_:  [int(an) for an in bfile2list(id_, max_seq_length=MAX_SEQ_LENGTH)]})]
                counter += 1
                if counter % PERIOD == 0:
                    timer(now, f"Scraping {counter} sequences ")
                    print(f"counter: {counter}")
                    df = pd.concat(to_concat, axis=1)
                    df.sort_index(axis=1).to_csv(csv_filename, index=False)
                    print(f"{counter} sequences written to csv")
                    print("check file: number of ids, file size?")


        # seqs_dict[idi] = bfile2list(idii, max_seq_length=100)
# pd.DataFrame(seqs_dict).sort_index(axis=1).to_csv(csv_filename, index=False)
df = pd.concat(to_concat, axis=1)
df.sort_index(axis=1).to_csv(csv_filename, index=False)
# # side effect are floats in csv, but maybe its unavoidable \_-.-_/
# magnitude = [f'{min(df[col]):e}  ' + f'{max(df[col]):e}' for col in df.columns]
# types = [type(df[col][0]) for col in df.columns]
# for i in magnitude:
#     print(i)
# for i in types:
#     print(i)

# # after download:
csv_filename = "linear_database.csv"
if os.getcwd()[-11:] == 'ProGED_oeis':
    csv_filename = "ProGED/examples/oeis/" + csv_filename
check = pd.read_csv(csv_filename)

print("Read file from csv:")
print(check)

# linseqs



timer(now)
# old_time, cpu_time = cpu_time, time.perf_counter()
# consumed = cpu_time - old_time
# print(f"\nScraping all (chosen) OEIS sequences"
#       f" took:\n {round(consumed, 1)} secconds,"
#       f" i.e. {round(consumed / 60, 2)} minutes"
#       f" or {round(consumed / 3600, 3)} hours.")
#


# to_conc = [{key: seq} for key, seq in linseqs.items()]
# cutoff = min(len(seq) for _, seq in seqs_dict.items())
# for id_, seq in seqs_dict.items():
#     print(id_, len(seq), seq)
# cutoff
# for orderii, idsii in linseqs.items():
#     for idii in idsii[:scale]:
#         seqs_dict[idii] = seqs_dict[idii][:cutoff]
#

# seqs_dict
# seqs_dict['A110164']
# seqs_dict['A350384']
# type(seqs_dict['A350384'])



# pd.DataFrame(seqs_dict)


#
# 1/0
#
#
# # soup.find_all('link')
# tag = soup.find_all('p')
# prices = soup.find_all(text='$')
# parent = prices[0].parent
# name = parent.name
# strong = parent.find('strong')
#
# specific_parent = prices[0].find_parent(class_='item-container')
#
# tag['color'] = "blue"
# print(tag.attrs)
# print(tag.attrs)
# for tag in tags:
#     print(tag.strip())  # '\n    $1233    ' -> '$123'
#
# # Search in all tags inside of the list:
# tags = soup.find_all(['option', 'div', 'li'], text='Undergraduate', value="undergraduate")
# tags = soup.find_all(class_='btn-value')
# tags = soup.find_all(text=re.compile("\$.*"))
#
# tbody = soup.tbody  # <tbody>
# trs = tbody.contents
# trs[0].next_sibling
# trs[0].previous_sibling
# trs[0].next_siblings
# trs[0].contents
# trs[0].descendents
#
#
# #
# # 1/0
# # movies = soup.find('table', {
# #     movies = soup.find('table', {
# #     'class': 'table'
# # }).find_all('a')
# # num = 0
# # for anchor in movies:
# #     urls = 'https://www.rottentomatoes.com' + anchor['href']
# # movies_lst.append(urls)
# # num += 1
# # movie_url = urls
# # movie_f = requests.get(movie_url, headers = headers)
# # movie_soup = BeautifulSoup(movie_f.content, 'lxml')
# # movie_content = movie_soup.find('div', {
# #     'class': 'movie_synopsis clamp clamp-6 js-clamp'
# # })
# # print(num, urls, '\n', 'Movie:' + anchor.string.strip())
# # print('Movie info:' + movie_content.string.strip())
# #
# #
