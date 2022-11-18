import requests
import lxml
from bs4 import BeautifulSoup
url = "https://www.rottentomatoes.com/top/bestofrt/"
headers = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'
}
f = requests.get(url, headers = headers)
# print(f.content)
# 1/0
movies_lst = []
soup = BeautifulSoup(f.content, 'lxml')
soup = BeautifulSoup(f.content, 'html.parser')
# soup.find_all('link')
tag = soup.find_all('p')
prices = soup.find_all(text='$')
parent = prices[0].parent
strong = parent.find('strong')

tag['color'] = "blue"
print(tag.attrs)
print(tag.attrs)
print(soup.prettify())
print(soup)
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
