# Запрос для поиска для HH & SO
qeury = 'python'
# Кол-во вакансий на одной странице для hh
items = 100

# Заголовок для сайта
headers = {
  'Host': 'hh.ru',
  'User-Agent': 'Safari',
  'Accept': '*/*',
  'Accept-Encoding': 'gzip, deflate, br',
  'Connection': 'keep-alive'
}
# URL - hh
url_hh = f'https://hh.ru/search/vacancy?items_on_page={items}&text={qeury}'
# URL - so
url_so = f'https://stackoverflow.com/jobs?id=158726&q={qeury}'


# https://hh.ru/search/vacancy?items_on_page=100&text=python
# https://hh.ru/search/vacancy?items_on_page=1&text=python&area=1002
# &area=1002 / Минск